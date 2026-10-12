"""Timeout-safe process boundary for Storm's native scheduler/model-checking API."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
import time

from .storm_results import StormError


SCHEDULER_BACKEND = "stormpy-native-scheduler-ovi-v1"


def stormpy_version() -> str:
    try:
        return version("stormpy")
    except PackageNotFoundError as error:
        raise StormError("Stormpy is required; reinstall the project with pip install -e '.[test]'") from error


def run_query_worker(request_path: Path, dimensions: int, timeout: float | None) -> tuple[list[dict], list[str]]:
    """One worker per oracle attempt; a fresh deadline for each native operation.

    A reader thread avoids a blocking readline on a crashed/stalled native
    solver, and drains stdout without mixing native logs into the protocol.
    """
    command = [sys.executable, "-m", "mdp_safety.storm_api_worker", str(request_path)]
    phases = ["ready", "support", "prepared"] + ["coordinate"] * dimensions
    log_path = request_path.with_suffix(".log")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=log, text=True, encoding="utf-8")
        messages: Queue = Queue()

        def read_messages():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = Thread(target=read_messages, daemon=True)
        reader.start()
        events = []
        try:
            previous = started
            for index, expected in enumerate(phases):
                try:
                    line = messages.get(timeout=timeout)
                except Empty as error:
                    raise StormError(f"Storm API timed out after {timeout:g}s during {expected}; see {log_path}") from error
                if line is None:
                    detail = log_path.read_text(encoding="utf-8")[-4000:]
                    raise StormError(f"Storm API worker exited before {expected}:\n{detail}")
                try:
                    event = json.loads(line)
                except (TypeError, ValueError) as error:
                    raise StormError("invalid Storm API worker response") from error
                if event.get("phase") == "error":
                    raise StormError(f"Storm API failed: {event['message']}; see {log_path}")
                if event.get("phase") != expected:
                    raise StormError(f"expected Storm API phase {expected}, got {event.get('phase')}")
                if expected == "coordinate" and event.get("dimension") != index - 3:
                    raise StormError("Storm API returned coordinates out of order")
                completed = float(event["completed_at"])
                event["elapsed_seconds"] = max(0., completed - previous)
                previous = completed
                events.append(event)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise StormError("Storm API worker timed out during shutdown") from error
            if process.returncode != 0:
                raise StormError(f"Storm API worker failed (exit {process.returncode}); see {log_path}")
            # Include worker teardown/IPC in the coordinate phase wall time.
            events[-1]["elapsed_seconds"] += max(0., time.perf_counter() - previous)
            return events, command
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            reader.join(timeout=5.)
            process.stdout.close()
