"""Finite explicit MDP data and semantics-preserving model transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class DrnError(ValueError):
    pass


@dataclass
class Action:
    identifier: str
    transitions: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class State:
    identifier: int
    labels: set[str] = field(default_factory=set)
    actions: list[Action] = field(default_factory=list)


@dataclass
class DrnModel:
    model_type: str
    states: list[State]

    @property
    def number_of_choices(self) -> int:
        return sum(len(state.actions) for state in self.states)

    @property
    def initial_state(self) -> int:
        initial = [state.identifier for state in self.states if "init" in state.labels]
        if len(initial) != 1:
            raise DrnError(f"expected exactly one fixed initial state, found {len(initial)}")
        return initial[0]

    def states_with_label(self, label: str) -> set[int]:
        return {state.identifier for state in self.states if label in state.labels}

    def validate(self, probability_tolerance: float = 1e-10) -> None:
        if self.model_type != "MDP":
            raise DrnError(f"expected an MDP, Storm exported {self.model_type!r}")
        identifiers = [state.identifier for state in self.states]
        if identifiers != list(range(len(self.states))):
            raise DrnError("DRN state identifiers must be contiguous and start at zero")
        self.initial_state
        for state in self.states:
            if not state.actions:
                raise DrnError(f"state {state.identifier} has no choices")
            for action in state.actions:
                if not action.transitions:
                    raise DrnError(
                        f"choice {action.identifier!r} of state {state.identifier} is empty"
                    )
                total = sum(probability for _, probability in action.transitions)
                if abs(total - 1.0) > probability_tolerance:
                    raise DrnError(
                        f"transition probabilities at state {state.identifier}, "
                        f"choice {action.identifier!r} sum to {total}"
                    )
                for successor, probability in action.transitions:
                    if successor < 0 or successor >= len(self.states):
                        raise DrnError(f"invalid successor state {successor}")
                    if probability < -probability_tolerance:
                        raise DrnError("negative transition probability")

    def with_derived_labels(self, definitions: list[dict[str, Any]]) -> "DrnModel":
        """Return a copy with labels defined by Boolean combinations of labels."""
        existing = set().union(*(state.labels for state in self.states))
        names: set[str] = set()
        compiled: list[tuple[str, set[str], set[str], set[str]]] = []
        for index, definition in enumerate(definitions):
            if not isinstance(definition, dict):
                raise DrnError(f"terminal_sets[{index}] must be an object")
            name = definition.get("name")
            if not isinstance(name, str) or not name:
                raise DrnError(f"terminal_sets[{index}].name must be a non-empty string")
            if name in existing or name in names:
                raise DrnError(f"derived terminal label {name!r} is not unique")
            predicates: list[set[str]] = []
            for key in ("all_of", "any_of", "none_of"):
                value = definition.get(key, [])
                if not isinstance(value, list) or not all(
                    isinstance(label, str) and label for label in value
                ):
                    raise DrnError(f"terminal_sets[{index}].{key} must be a string array")
                predicates.append(set(value))
            all_of, any_of, none_of = predicates
            if not all_of and not any_of and not none_of:
                raise DrnError(
                    f"terminal_sets[{index}] needs at least one label predicate"
                )
            missing = (all_of | any_of | none_of) - existing
            if missing:
                raise DrnError(
                    f"terminal_sets[{index}] references unknown labels: {sorted(missing)}"
                )
            names.add(name)
            compiled.append((name, all_of, any_of, none_of))
        states: list[State] = []
        for state in self.states:
            labels = set(state.labels)
            for name, all_of, any_of, none_of in compiled:
                matches = all_of <= state.labels and (
                    not any_of or bool(any_of & state.labels)
                )
                if matches and not (none_of & state.labels):
                    labels.add(name)
            states.append(
                State(
                    state.identifier,
                    labels,
                    [
                        Action(action.identifier, list(action.transitions))
                        for action in state.actions
                    ],
                )
            )
        result = DrnModel(self.model_type, states)
        result.validate()
        return result

    def validate_terminal_labels(
        self, labels: list[str], require_absorbing: bool = True
    ) -> dict[int, int]:
        if not labels:
            raise DrnError("at least one terminal label is required")
        if len(set(labels)) != len(labels):
            raise DrnError("terminal labels must be unique")
        terminal_to_dimension: dict[int, int] = {}
        for dimension, label in enumerate(labels):
            matched = self.states_with_label(label)
            if not matched:
                raise DrnError(f"terminal label {label!r} matches no reachable state")
            for state_id in matched:
                if state_id in terminal_to_dimension:
                    previous = labels[terminal_to_dimension[state_id]]
                    raise DrnError(
                        f"terminal labels {previous!r} and {label!r} overlap at state {state_id}"
                    )
                terminal_to_dimension[state_id] = dimension
        for state_id in terminal_to_dimension:
            if not require_absorbing:
                continue
            state = self.states[state_id]
            for action in state.actions:
                self_probability = sum(
                    probability
                    for successor, probability in action.transitions
                    if successor == state_id
                )
                if abs(self_probability - 1.0) > 1e-10:
                    raise DrnError(
                        f"state {state_id} is labeled terminal but is not absorbing"
                    )
        return terminal_to_dimension

    def global_choice_map(self) -> dict[int, tuple[int, int]]:
        result: dict[int, tuple[int, int]] = {}
        choice = 0
        for state in self.states:
            for local_index, _ in enumerate(state.actions):
                result[choice] = (state.identifier, local_index)
                choice += 1
        return result

    def weighted_terminal_gadget(
        self,
        terminal_to_dimension: dict[int, int],
        weights: Iterable[float],
        success_label: str,
        failure_label: str,
    ) -> "DrnModel":
        values = list(float(value) for value in weights)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise DrnError("gadget weights must lie in [0, 1]")
        all_labels = set().union(*(state.labels for state in self.states))
        if success_label in all_labels or failure_label in all_labels:
            raise DrnError("reserved gadget label already occurs in the model")
        success = len(self.states)
        failure = success + 1
        states: list[State] = []
        for original in self.states:
            if original.identifier in terminal_to_dimension:
                weight = values[terminal_to_dimension[original.identifier]]
                transitions: list[tuple[int, float]] = []
                if weight > 0.0:
                    transitions.append((success, weight))
                if weight < 1.0:
                    transitions.append((failure, 1.0 - weight))
                actions = [Action("0", transitions)]
            else:
                actions = [
                    Action(action.identifier, list(action.transitions))
                    for action in original.actions
                ]
            states.append(State(original.identifier, set(original.labels), actions))
        states.extend(
            (
                State(success, {success_label}, [Action("0", [(success, 1.0)])]),
                State(failure, {failure_label}, [Action("0", [(failure, 1.0)])]),
            )
        )
        model = DrnModel("MDP", states)
        model.validate()
        return model
