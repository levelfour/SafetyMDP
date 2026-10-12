"""Native scheduler semantics, sound bounds, worker deadlines and provenance."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from mdp_safety.model import Action, DrnModel, State
from mdp_safety.storm import StormRunner
from mdp_safety.storm_api import SCHEDULER_BACKEND, run_query_worker
from mdp_safety.storm_results import StormError


HAS_STORM = shutil.which("storm") is not None


def cyclic_model():
    return DrnModel("MDP", [
        State(0, {"init"}, [
            Action("left", [(0, .5), (1, .25), (2, .125), (3, .125)]),
            Action("right", [(0, .5), (1, .05), (2, .35), (3, .10)]),
        ]),
        State(1, {"left"}, [Action("absorb", [(1, 1.)])]),
        State(2, {"right"}, [Action("absorb", [(2, 1.)])]),
        State(3, {"other"}, [Action("loop", [(3, 1.)])]),
        # Unreachable state with multiple choices exercises native completion.
        State(4, set(), [Action("a", [(1, 1.)]), Action("b", [(2, 1.)])]),
    ])


@pytest.mark.skipif(not HAS_STORM, reason="Storm is not installed")
@pytest.mark.parametrize("weight,expected", [([1., 0.], [.5, .25]), ([0., 1.], [.1, .7]),
                                           ([.5, .5], [.1, .7])])
def test_native_scheduler_coordinates_match_exact_cyclic_solution(tmp_path, weight, expected):
    import stormpy
    model = cyclic_model()
    runner = StormRunner(timeout_seconds=None)
    query = runner.query(model, {1: 0, 2: 1}, ["left", "right"], np.array(weight), 1e-9, tmp_path, "cyclic")
    for coordinate, exact in zip(query.coordinates, expected):
        assert coordinate.lower <= exact <= coordinate.upper
        assert coordinate.estimate == pytest.approx(exact, abs=1e-9)
    optimum = float(np.dot(weight, expected))
    assert query.support.interval.lower <= optimum <= query.support.interval.upper
    assert query.support.scheduler_actions[0] == ("left" if weight == [1., 0.] else "right")
    assert 4 in query.support.scheduler_actions
    assert runner.configuration["backend"] == SCHEDULER_BACKEND
    assert runner.configuration["stormpy_storm_version"]
    induced = stormpy.build_model_from_drn(str(tmp_path / "induced-cyclic.drn"))
    assert induced.model_type == stormpy.ModelType.DTMC
    assert induced.nr_states == len(model.states) + 2
    assert induced.nr_choices == induced.nr_states
    # The coordinate optimum of the other action must NOT leak into this policy.
    assert sum(c.estimate for c in query.coordinates) < .9


@pytest.mark.skipif(not HAS_STORM, reason="Storm is not installed")
@pytest.mark.parametrize("initial_target", [False, True])
def test_native_coordinates_preserve_first_hit_and_initial_targets(tmp_path, initial_target):
    model = DrnModel("MDP", [
        State(0, set() if initial_target else {"init"}, [Action("go", [(1, 1.)])]),
        State(1, {"a", "init"} if initial_target else {"a"}, [Action("leave", [(2, 1.)])]),
        State(2, {"b"}, [Action("stay", [(2, 1.)])]),
    ])
    query = StormRunner().query(model, {1: 0, 2: 1}, ["a", "b"], np.array([.2, .8]),
                                1e-8, tmp_path, "first-hit")
    assert [c.estimate for c in query.coordinates] == pytest.approx([1., 0.])
    assert query.support.interval.estimate == pytest.approx(.2)


@pytest.mark.skipif(not HAS_STORM, reason="Storm is not installed")
def test_multiple_target_states_and_nontermination(tmp_path):
    model = DrnModel("MDP", [
        State(0, {"init"}, [Action("go", [(1, .1), (2, .2), (3, .3), (4, .4)])]),
        *[State(i, {"a" if i < 3 else "b"} if i < 4 else set(),
                [Action("loop", [(i, 1.)])]) for i in range(1, 5)],
    ])
    query = StormRunner().query(model, {1: 0, 2: 0, 3: 1}, ["a", "b"], np.array([.7, 1.]),
                                1e-9, tmp_path, "multiple")
    for coordinate in query.coordinates:
        assert coordinate.lower <= .3 <= coordinate.upper


@pytest.mark.skipif(not HAS_STORM, reason="Storm is not installed")
@pytest.mark.parametrize("family", ["gridworld", "robot-mission", "network-routing"])
@pytest.mark.parametrize("weight", [[1., 1., 1., 1.], [1., .6, .2, 0.]])
def test_paper_d4_scheduler_matches_original_model_linear_equations(tmp_path, family, weight):
    root = Path(__file__).resolve().parents[1]
    runner = StormRunner(timeout_seconds=30.)
    model = runner.build_prism(root / f"benchmarks/models/{family}-d4.prism", tmp_path / "model.drn")
    labels = json.loads((root / f"benchmarks/specs/{family}-d4-safe.json").read_text())["terminal_labels"]
    terminals = model.validate_terminal_labels(labels)
    query = runner.query(model, terminals, labels, np.array(weight), 1e-8, tmp_path, "paper")
    # Independent reference on the ORIGINAL model, using the exported action
    # map, not Storm's induced model or the implementation's model transformation.
    matrix = np.zeros((len(model.states), len(model.states)))
    for state in model.states:
        if state.identifier in terminals:
            matrix[state.identifier, state.identifier] = 1.
        else:
            choice = query.support.scheduler[state.identifier]
            for successor, probability in state.actions[choice].transitions:
                matrix[state.identifier, successor] += probability
    for label, interval in zip(labels, query.coordinates):
        target = model.states_with_label(label)
        reachable = set(target)
        while True:
            predecessors = {s for s in range(len(model.states))
                            if any(matrix[s, t] > 0 for t in reachable)}
            updated = reachable | predecessors
            if updated == reachable:
                break
            reachable = updated
        unknown = sorted(reachable - target)
        values = np.zeros(len(model.states))
        values[list(target)] = 1.
        if unknown:
            values[unknown] = np.linalg.solve(np.eye(len(unknown)) - matrix[np.ix_(unknown, unknown)],
                                             matrix[np.ix_(unknown, sorted(target))].sum(axis=1))
        exact = values[model.initial_state]
        assert interval.lower - 1e-13 <= exact <= interval.upper + 1e-13
        assert interval.estimate == pytest.approx(exact, abs=1e-8)


def worker_stub(monkeypatch, code):
    from mdp_safety import storm_api
    original = subprocess.Popen
    processes = []

    def popen(command, **kwargs):
        process = original([sys.executable, "-u", "-c", code], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(storm_api.subprocess, "Popen", popen)
    return processes


def test_native_worker_timeout_terminates_and_reaps_process(tmp_path, monkeypatch):
    processes = worker_stub(monkeypatch, "import time; time.sleep(10)")
    with pytest.raises(StormError, match="timed out.*ready"):
        run_query_worker(tmp_path / "request.json", 2, .1)
    assert processes[0].poll() is not None


@pytest.mark.parametrize("limit", [None, .8])
def test_worker_deadline_restarts_for_each_coordinate(tmp_path, monkeypatch, limit):
    code = """import json, time
for message in [{'phase':'ready'}, {'phase':'support'}, {'phase':'prepared'},
                {'phase':'coordinate','dimension':0}, {'phase':'coordinate','dimension':1}]:
    time.sleep(.2)
    message['completed_at'] = time.perf_counter()
    print(json.dumps(message), flush=True)
"""
    processes = worker_stub(monkeypatch, code)
    events, _ = run_query_worker(tmp_path / "request.json", 2, limit)
    assert [e["phase"] for e in events] == ["ready", "support", "prepared", "coordinate", "coordinate"]
    assert all(e["elapsed_seconds"] >= 0 for e in events)
    assert processes[0].returncode == 0


def test_interruption_reaps_native_worker(tmp_path, monkeypatch):
    from mdp_safety import storm_api
    processes = worker_stub(monkeypatch, "import time; time.sleep(10)")
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(storm_api.Queue, "get", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_query_worker(tmp_path / "request.json", 2, None)
    assert processes[0].poll() is not None


@pytest.mark.parametrize("code,match", [
    ("print('not json')", "invalid Storm API"),
    ("raise SystemExit(4)", "exited before ready"),
    ("print('{\"phase\":\"error\",\"message\":\"solver failed\"}')", "solver failed"),
    ("print('{\"phase\":\"prepared\"}')", "expected Storm API phase ready"),
])
def test_worker_failures_are_not_probabilities(tmp_path, monkeypatch, code, match):
    processes = worker_stub(monkeypatch, code)
    with pytest.raises(StormError, match=match):
        run_query_worker(tmp_path / "request.json", 1, 2.)
    assert processes[0].poll() is not None


def test_old_backend_cannot_be_resumed(tmp_path):
    from mdp_safety.experiments.checkpoint import _configuration, _validate_resume
    from mdp_safety.experiments.models import BenchmarkSettings
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"name": "test", "model": "m.pm", "specification": "s.json"}]))
    settings = BenchmarkSettings(manifest, tmp_path)
    config = _configuration(settings, state="finished", started_at="test", benchmark_wall_seconds=0.,
                            jobs=1, completed_jobs=1)
    assert config.pop("oracle_backend") == SCHEDULER_BACKEND
    (tmp_path / "benchmark_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="oracle_backend"):
        _validate_resume(settings)


def test_native_worker_reuses_one_induced_model_and_explicit_ovi_settings(tmp_path, monkeypatch):
    import stormpy
    from mdp_safety.drn import write_drn
    from mdp_safety.storm_api_worker import run_query
    model = cyclic_model().weighted_terminal_gadget({1: 0, 2: 1}, [1., 0.], "success", "failure")
    path = tmp_path / "weighted.drn"
    write_drn(model, path)
    calls, settings = [], []
    check, configure = stormpy.model_checking, stormpy.set_settings

    def record_check(model, prop, **kwargs):
        calls.append((model, kwargs))
        return check(model, prop, **kwargs)

    def record_settings(args):
        settings.extend(args)
        configure(args)

    monkeypatch.setattr(stormpy, "model_checking", record_check)
    monkeypatch.setattr(stormpy, "set_settings", record_settings)
    events = []
    run_query({
        "model_path": str(path), "support_property": 'Pmax=? [F "success"]', "precision": 1e-9,
        "scheduler_path": str(tmp_path / "scheduler.json"),
        "support_result_path": str(tmp_path / "support.json"),
        "induced_path": str(tmp_path / "induced.drn"), "terminal_labels": ["left", "right"],
        "coordinate_result_paths": [str(tmp_path / f"coordinate-{i}.json") for i in range(2)],
    }, events.append)
    assert len(calls) == 3
    assert calls[0][1]["extract_scheduler"] is True
    assert calls[1][0] is calls[2][0]
    assert calls[1][0].model_type == stormpy.ModelType.DTMC
    assert all(not kwargs.get("extract_scheduler", False) for _, kwargs in calls[1:])
    assert all(kwargs["environment"] is calls[0][1]["environment"] for _, kwargs in calls)
    solver = calls[0][1]["environment"].solver_environment
    assert solver.minmax_solver_environment.method == stormpy.MinMaxMethod.optimistic_value_iteration
    assert solver.native_solver_environment.method == stormpy.NativeLinearEquationSolverMethod.optimistic_value_iteration
    assert float(solver.minmax_solver_environment.precision) == pytest.approx(1e-9)
    assert float(solver.native_solver_environment.precision) == pytest.approx(1e-9)
    assert {"--sound", "--minmax:absolute", "--native:absolute"} <= set(settings)
