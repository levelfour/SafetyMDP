"""Isolated Stormpy query: synthesize once, apply that scheduler, check all targets.

Run only through storm_api.run_query_worker. A fresh process isolates Storm's
global settings and makes even a native C++ call interruptible by the parent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback


def run_query(request: dict, emit) -> None:
    import stormpy
    import stormpy.info

    precision = float(request["precision"])
    # Construct the Environment AFTER setting the absolute-error options;
    # model_checking's default Environment was created at import time.
    stormpy.set_settings([
        "--sound", "--minmax:method", "optimistic-value-iteration",
        "--minmax:absolute", "--minmax:precision", format(precision, ".17g"),
        "--native:method", "optimistic-value-iteration",
        "--native:absolute", "--native:precision", format(precision, ".17g"),
    ])
    environment = stormpy.Environment()
    environment.solver_environment.set_force_sound(True)
    environment.solver_environment.set_linear_equation_solver_type(stormpy.EquationSolverType.native)
    model = stormpy.build_model_from_drn(request["model_path"])
    if len(model.initial_states) != 1:
        raise ValueError("expected one fixed initial state")
    initial = model.initial_states[0]
    emit({"phase": "ready", "stormpy_version": stormpy.__version__,
          "storm_version": stormpy.info.storm_version()})

    support_property = stormpy.parse_properties(request["support_property"])[0]
    result = stormpy.model_checking(model, support_property, only_initial_states=True,
                                   extract_scheduler=True, environment=environment)
    if not result.has_scheduler:
        raise ValueError("Storm did not return a scheduler")
    scheduler = result.scheduler
    if not scheduler.memoryless or not scheduler.deterministic:
        raise ValueError("expected a deterministic memoryless Storm scheduler")
    # Complete don't-care states using Storm's scheduler API, then evaluate
    # exactly this completed object, not a reconstructed action table.
    for state in model.states:
        choice = scheduler.get_choice(state.id)
        if not choice.defined:
            scheduler.set_choice(stormpy.SchedulerChoice(0), state.id)
        elif choice.get_deterministic_choice() >= len(state.actions):
            raise ValueError(f"invalid scheduler choice at state {state.id}")
    if scheduler.partial:
        raise ValueError("Storm scheduler is still partial after completion")
    Path(request["scheduler_path"]).write_text(scheduler.to_json_str(model), encoding="utf-8")
    estimate = float(result.at(initial))
    _write_value(request["support_result_path"], initial, estimate)
    emit({"phase": "support", "estimate": estimate})

    # Target states in the weighted gadget lead only to the two absorbing
    # auxiliary sinks. Hence reaching each original target has exactly the
    # same probability as in the original MDP stopped at its first target.
    # Applying the actual native scheduler to this same model avoids all
    # state/action remapping, including scheduler transfers across models.
    induced = model.apply_scheduler(scheduler, drop_unreachable_states=False)
    if induced.model_type != stormpy.ModelType.DTMC:
        raise ValueError("a complete deterministic scheduler did not induce a DTMC")
    if list(induced.initial_states) != [initial]:
        raise ValueError("scheduler application changed initial state numbering")
    export_options = stormpy.DirectEncodingExporterOptions()
    export_options.outputPrecision = 17
    stormpy.export_to_drn(induced, request["induced_path"], export_options)
    emit({"phase": "prepared"})

    for dimension, label in enumerate(request["terminal_labels"]):
        property_text = f"P=? [F {json.dumps(label)}]"
        formula = stormpy.parse_properties(property_text)[0]
        result = stormpy.model_checking(induced, formula, only_initial_states=True,
                                       environment=environment)
        estimate = float(result.at(initial))
        _write_value(request["coordinate_result_paths"][dimension], initial, estimate)
        emit({"phase": "coordinate", "dimension": dimension, "estimate": estimate})


def _write_value(path: str, initial: int, estimate: float) -> None:
    Path(path).write_text(json.dumps([{"s": initial, "l": ["init"], "v": estimate}],
                                    allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    # Keep protocol messages on their own descriptor. Native Storm logging
    # (including C++ stdout) goes to the log file attached as stderr.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as protocol:
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        def emit(message):
            message["completed_at"] = time.perf_counter()
            protocol.write(json.dumps(message, allow_nan=False) + "\n")
            protocol.flush()

        try:
            request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
            run_query(request, emit)
        except Exception as error:
            traceback.print_exc()
            emit({"phase": "error", "message": f"{type(error).__name__}: {error}"})
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
