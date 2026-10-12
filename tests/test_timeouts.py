"""Optional deadlines propagate without substituting a large finite timeout."""

from dataclasses import replace
import json
import signal
import subprocess
import sys

import pytest

from mdp_safety import benchmark, cli, storm
from mdp_safety.benchmark_report import SuiteArtifacts, render_report
from mdp_safety.experiments import execution
from mdp_safety.experiments.checkpoint import _configuration, _validate_resume
from mdp_safety.experiments.models import BenchmarkSettings
from mdp_safety.timeouts import format_timeout, parse_timeout, validate_timeout
from mdp_safety.verification.options import VerificationOptions


@pytest.mark.parametrize("value,expected", [("none", None), (" NONE ", None), ("200", 200.), ("0.5", .5)])
def test_parse_explicit_optional_timeout(value, expected):
    assert parse_timeout(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "inf", "nan", "", "unlimited"])
def test_invalid_cli_timeout_is_not_silently_disabled(value):
    with pytest.raises(ValueError):
        parse_timeout(value)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "none"])
def test_invalid_api_timeout_is_rejected(value, tmp_path):
    with pytest.raises(ValueError):
        validate_timeout(value)
    with pytest.raises(ValueError):
        VerificationOptions(storm_timeout_seconds=value).validate()
    with pytest.raises(ValueError):
        BenchmarkSettings(tmp_path / "m.json", tmp_path, run_timeout_seconds=value).validate()


def test_none_is_valid_for_both_settings_and_cli_parsers(tmp_path):
    VerificationOptions(storm_timeout_seconds=None, max_iterations=50).validate()
    BenchmarkSettings(tmp_path / "m.json", tmp_path, storm_timeout_seconds=None,
                      run_timeout_seconds=None, max_iterations=50).validate()
    single = cli._parser().parse_args(["m.pm", "s.json", "--timeout", "none", "--max-iterations", "50"])
    batch = benchmark._parser().parse_args([
        "m.json", "--output-dir", str(tmp_path), "--timeout", "none",
        "--run-timeout", "none", "--max-iterations", "50"])
    assert single.timeout is batch.timeout is batch.run_timeout is None
    assert single.max_iterations == batch.max_iterations == 50
    assert cli._parser().parse_args(["m.pm", "s.json"]).timeout == 300
    assert format_timeout(None) == "disabled"
    assert format_timeout(200.) == "200 s"


@pytest.mark.parametrize("limit", [None, 200.])
def test_storm_passes_optional_timeout_to_subprocess(monkeypatch, limit):
    calls = []
    monkeypatch.setattr(storm.shutil, "which", lambda _: sys.executable)

    def fake_run(command, **kwargs):
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, "Storm 1.13.0", "")

    monkeypatch.setattr(storm.subprocess, "run", fake_run)
    runner = storm.StormRunner(timeout_seconds=limit)
    runner._run(["fake-query"])
    assert calls == [limit, limit]


@pytest.mark.parametrize("limit", [None, 5.])
def test_execute_command_finishes_normally_with_optional_deadline(limit):
    result, timed_out = execution.execute_command([sys.executable, "-c", "print('done')"], limit)
    assert not timed_out and result.returncode == 0
    assert result.stdout.strip() == "done"


@pytest.mark.parametrize("limit", [None, 200.])
@pytest.mark.parametrize("force_kill", [False, True])
def test_interruption_reaps_verifier_process_group(monkeypatch, limit, force_kill):
    if execution.os.name != "posix":
        pytest.skip("POSIX process-group contract")
    signals = []
    waits = []

    class FakeProcess:
        pid = 123456

        def communicate(self, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise KeyboardInterrupt
            if len(waits) == 2 and force_kill:
                raise subprocess.TimeoutExpired("fake", timeout)
            return "", ""

    def popen(*args, **kwargs):
        assert kwargs["start_new_session"]
        return FakeProcess()

    monkeypatch.setattr(execution.subprocess, "Popen", popen)
    monkeypatch.setattr(execution.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        execution.execute_command(["not-a-real-process"], limit)
    assert waits == ([limit, 5., None] if force_kill else [limit, 5.])
    expected = [signal.SIGTERM, signal.SIGKILL] if force_kill else [signal.SIGTERM]
    assert [sig for _, sig in signals] == expected
    assert all(pid == FakeProcess.pid for pid, _ in signals)


def test_finite_whole_run_timeout_still_terminates():
    result, timed_out = execution.execute_command([sys.executable, "-c", "import time; time.sleep(10)"], .05)
    assert timed_out and result.returncode == 124


def test_null_timeouts_are_persisted_rendered_and_compared_for_resume(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"name": "tiny", "model": "m.pm", "specification": "s.json"}]))
    settings = BenchmarkSettings(manifest, tmp_path, storm_timeout_seconds=None,
                                 run_timeout_seconds=None, max_iterations=50)
    config = _configuration(settings, state="finished", started_at="test", benchmark_wall_seconds=1.,
                            jobs=1, completed_jobs=1)
    (tmp_path / "benchmark_config.json").write_text(json.dumps(config))
    assert config["timeout_seconds"] is config["run_timeout_seconds"] is None
    _validate_resume(settings)
    for field, value in [("storm_timeout_seconds", 200.), ("run_timeout_seconds", 200.), ("max_iterations", 100)]:
        with pytest.raises(ValueError, match="changed settings"):
            _validate_resume(replace(settings, **{field: value}))
    report = render_report([SuiteArtifacts("Domains", tmp_path, config, (), (), ())], "Study")
    assert "Storm-call timeout: disabled; whole-run timeout: disabled" in report
    assert "max iterations: 50" in report
    assert "None s" not in report
