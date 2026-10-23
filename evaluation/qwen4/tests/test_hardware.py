import json
import subprocess

from hardware import inspect_hardware


def test_gpu_probe_is_a_separate_process(monkeypatch):
    calls = []

    def probe(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"hardware": [{"name": "MI308X"}]}), ""
        )

    monkeypatch.setattr("hardware.subprocess.run", probe)
    assert inspect_hardware(4)["hardware"][0]["name"] == "MI308X"
    assert calls[0][-2:] == ["--gpus", "4"]
    assert calls[0][1].endswith("hardware.py")
