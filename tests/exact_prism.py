"""Small exact Storm export reader shared by fixed-input regression tests."""
from fractions import Fraction
import re
import shlex
import shutil
import subprocess

STORM = shutil.which("storm")


def read_exact_drn(path):
    states = {}
    state = action = None
    text = path.read_text()
    assert "@value_type: rational" in text
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("state "):
            fields = shlex.split(line)
            state = {"labels": frozenset(fields[2:]), "actions": {}}
            states[int(fields[1])] = state
        elif line.startswith("//["):
            state["valuation"] = {
                k: {"true": 1, "false": 0}.get(v, v)
                for k, v in re.findall(r"(\w+)=(true|false|-?\d+)", line)
            }
            state["valuation"] = {k: int(v) for k, v in state["valuation"].items()}
            for token in line[3:-1].split("&"):
                token = token.strip()
                if "=" not in token:
                    assert re.fullmatch(r"!?\w+", token)
                    state["valuation"][token.lstrip("!")] = int(not token.startswith("!"))
        elif line.startswith("action "):
            name = shlex.split(line)[1]
            assert name not in state["actions"]
            action = {}
            state["actions"][name] = action
        elif state is not None and ":" in line and not line.startswith("//"):
            destination, value = line.split(":", 1)
            target = int(destination)
            action[target] = action.get(target, Fraction(0)) + Fraction(value.strip())
    assert states and all("valuation" in state for state in states.values())
    return states


def export_exact(prism, destination):
    subprocess.run([STORM, "--prism", str(prism), "--exact", "--buildfull",
                    "--buildstateval", "--buildchoicelab", "--explchecks", "--nofixdl",
                    "--exportbuild", str(destination), "drn"],
                   check=True, capture_output=True, text=True, timeout=60)
    return read_exact_drn(destination)
