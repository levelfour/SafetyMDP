import pytest

from mdp_safety.drn import DrnError, read_drn, write_drn


def test_read_validate_and_roundtrip_explicit_model(tmp_path) -> None:
    # Use Storm's exported equivalent kept deliberately small in this fixture.
    source = tmp_path / "model.drn"
    source.write_text(
        """@type: MDP
@value_type: double
@nr_states
3
@nr_choices
4
@model
state 0 init
 action 0
  1 : 0.5
  2 : 0.5
 action 1
  1 : 1
state 1 \"a terminal\"
 action 0
  1 : 1
state 2 b
 action 0
  2 : 1
""",
        encoding="utf-8",
    )
    model = read_drn(source)
    terminal_map = model.validate_terminal_labels(["a terminal", "b"])
    weighted = model.weighted_terminal_gadget(
        terminal_map, [0.25, 0.75], "success", "failure"
    )
    destination = tmp_path / "weighted.drn"
    write_drn(weighted, destination)
    reparsed = read_drn(destination)
    assert len(reparsed.states) == 5
    assert reparsed.states_with_label("success") == {3}


def test_overlapping_terminal_labels_are_rejected(tmp_path) -> None:
    path = tmp_path / "overlap.drn"
    path.write_text(
        """@type: MDP
@nr_states
1
@model
state 0 init a b
 action 0
  0 : 1
""",
        encoding="utf-8",
    )
    with pytest.raises(DrnError, match="overlap"):
        read_drn(path).validate_terminal_labels(["a", "b"])


def test_derived_terminal_sets_and_first_hit_terminalization(tmp_path) -> None:
    path = tmp_path / "derived.drn"
    path.write_text(
        """@type: MDP
@nr_states
3
@model
state 0 init
 action 0
  1 : 1
state 1 finished agree zero
 action 0
  2 : 1
state 2 sink
 action 0
  2 : 1
""",
        encoding="utf-8",
    )
    model = read_drn(path).with_derived_labels(
        [
            {"name": "decision_zero", "all_of": ["finished", "zero"]},
            {"name": "disagree", "all_of": ["finished"], "none_of": ["agree"]},
        ]
    )
    with pytest.raises(DrnError, match="not absorbing"):
        model.validate_terminal_labels(["decision_zero"], require_absorbing=True)
    terminal_map = model.validate_terminal_labels(
        ["decision_zero"], require_absorbing=False
    )
    weighted = model.weighted_terminal_gadget(terminal_map, [1.], "success", "failure")
    assert weighted.states[1].actions[0].transitions == [(3, 1.0)]
    assert weighted.states[3].actions[0].transitions == [(3, 1.0)]


def test_derived_terminal_set_can_use_only_none_of(tmp_path) -> None:
    path = tmp_path / "complement.drn"
    path.write_text(
        """@type: MDP
@nr_states
2
@model
state 0 init good
 action 0
  1 : 1
state 1 bad
 action 0
  1 : 1
""",
        encoding="utf-8",
    )
    model = read_drn(path).with_derived_labels(
        [{"name": "not_good", "none_of": ["good"]}]
    )
    assert model.states_with_label("not_good") == {1}
