"""Strategy catalog, unsupported-name rejection and fail-fast configuration tests."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from mdp_safety import cli, projections
from mdp_safety.benchmark_report import _method
from mdp_safety.experiments.manifest import load_manifest
from mdp_safety.experiments.models import BenchmarkSettings, DEFAULT_PROJECTIONS
from mdp_safety.experiments.runner import run_benchmark
from mdp_safety.strategies import registry
from mdp_safety.verifier import SafetyVerifier, VerificationOptions


ROOT = Path(__file__).resolve().parents[1]


def test_builtin_catalog_is_the_shared_source_of_defaults_and_labels():
    assert DEFAULT_PROJECTIONS is registry.DEFAULT_PROJECTIONS
    assert DEFAULT_PROJECTIONS == ("l2", "linf", "gauge", "sandwich-lk", "sandwich-lk-gauge")
    assert registry.COMPARISON_PROJECTIONS == (
        "l2", "linf", "gauge", "sandwich-lk", "sandwich-lk-gauge",
    )
    assert set(registry.projection_names()) == {item.name for item in registry.BUILTIN_STRATEGIES}
    for item in registry.BUILTIN_STRATEGIES:
        assert isinstance(registry.make_projection(item.name), item.factory().__class__)
        assert _method(item.name) == item.label
        assert item.description in registry.projection_help()


def test_dc_has_no_module_factory_or_public_export():
    assert importlib.util.find_spec("mdp_safety.strategies.dc") is None
    assert not hasattr(projections, "DcSupportGapStrategy")
    with pytest.raises(ValueError, match="unknown projection"):
        registry.make_projection("dc")
    with pytest.raises(ValueError, match="unknown projection"):
        SafetyVerifier(VerificationOptions(projection="dc"))
    assert _method("unregistered") == "unregistered"


def test_cli_rejects_dc_before_opening_inputs(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["missing.pm", "missing.json", "--projection", "dc"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("values", [
    ["dc"], ["sandwich"], ["sandwich-l2"], ["missing"], ["l2", "l2"], [None], [], "l2", None,
])
def test_invalid_manifest_methods_fail_before_executor_and_output(tmp_path, values):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{
        "name": "tiny", "model": "model.pm", "specification": "safe.json",
        "projections": values,
    }]))
    output = tmp_path / "absent"

    def forbidden(*args):
        pytest.fail("invalid configuration must not reach the executor")

    with pytest.raises(ValueError, match="projections"):
        run_benchmark(BenchmarkSettings(manifest, output), executor=forbidden)
    assert not output.exists()


def test_plugin_factory_key_errors_are_not_misreported(monkeypatch):
    def broken():
        raise KeyError("plugin-internal-key")
    monkeypatch.setitem(registry._STRATEGIES, "test-broken", broken)
    with pytest.raises(KeyError, match="plugin-internal-key"):
        registry.make_projection("test-broken")
    assert registry.make_projection(" L2 ").name == "l2"


def test_all_shipped_profiles_and_manifests_use_supported_strategies(tmp_path):
    from scripts.run_study import plan
    paths = [ROOT / "experiments/paper.json", *(ROOT / "experiments/manifests").glob("*.json")]
    for path in paths:
        data = json.loads(path.read_text())
        if "methods" in data:
            registry.validate_projection_names(data["methods"])
            suites = plan(ROOT, tmp_path / "absent", data, "missing-storm", False)
            assert {suite.key for suite in suites} == {"domains"}
        else:
            for instance in load_manifest(path):
                registry.validate_projection_names(instance.projections)
                assert instance.model.is_file() and instance.specification.is_file(), path
        for retired in ("risk_scale", "fm26", "qvbs", "sound-mopmc"):
            assert retired not in path.read_text().lower(), path
    assert not (tmp_path / "absent").exists()
    for relative in ("domains.json", "domains_safe.json", "domains_unsafe.json"):
        for instance in load_manifest(ROOT / "experiments/manifests" / relative):
            registry.validate_projection_names(instance.projections)


@pytest.mark.parametrize("verification_field,benchmark_field,value", [
    ("geometry_tolerance", "geometry_tolerance", float("nan")),
    ("geometry_tolerance", "geometry_tolerance", float("inf")),
    ("geometry_tolerance", "geometry_tolerance", 0),
    ("storm_timeout_seconds", "storm_timeout_seconds", float("nan")),
    ("storm_timeout_seconds", "storm_timeout_seconds", float("inf")),
    ("storm_timeout_seconds", "storm_timeout_seconds", 0),
    ("max_iterations", "max_iterations", 1.5),
    ("max_iterations", "max_iterations", True),
    ("storm_precision", "precision", 0),
    ("minimum_storm_precision", "minimum_precision", float("nan")),
])
def test_verifier_and_benchmark_share_numeric_validation(
    tmp_path, verification_field, benchmark_field, value,
):
    with pytest.raises(ValueError):
        replace(VerificationOptions(), **{verification_field: value}).validate()
    with pytest.raises(ValueError):
        replace(BenchmarkSettings(tmp_path / "m.json", tmp_path),
                **{benchmark_field: value}).validate()


@pytest.mark.parametrize("field,value", [
    ("run_timeout_seconds", float("nan")), ("run_timeout_seconds", float("inf")),
    ("repetitions", True), ("repetitions", 1.5), ("repetitions", 0),
])
def test_benchmark_only_limits_are_validated(tmp_path, field, value):
    with pytest.raises(ValueError):
        replace(BenchmarkSettings(tmp_path / "m.json", tmp_path), **{field: value}).validate()

@pytest.mark.parametrize("name,export", [
    ("sandwich", "SandwichStrategy"), ("sandwich-l2", "SandwichL2Strategy"),
])
def test_retired_sandwich_methods_are_rejected_not_aliased(name, export, capsys):
    from mdp_safety.strategies import sandwich
    assert not hasattr(sandwich, export)
    assert not hasattr(projections, export)
    assert importlib.util.find_spec("mdp_safety.inner_projection") is None
    with pytest.raises(ValueError, match="unknown projection"):
        registry.make_projection(name)
    with pytest.raises(ValueError, match="unknown projection"):
        SafetyVerifier(VerificationOptions(projection=name))
    with pytest.raises(SystemExit) as error:
        cli.main(["missing.pm", "missing.json", "--projection", name])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
