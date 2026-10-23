import json

import pytest
from common import write_json
from config import make_plan
from docker_backend import DockerBackend


def test_one_build_and_distinct_container_per_pair(tmp_path, monkeypatch):
    calls = []

    class Process:
        def __init__(self, argv, log_path):
            calls.append(argv)

        def wait(self, timeout, progress=False):
            return 0

    monkeypatch.setattr("docker_backend.ManagedProcess", Process)
    plan = make_plan(run_id="test-lifecycle")
    backend = DockerBackend(plan, tmp_path, models_root="/raid/models")

    def run(argv):
        calls.append(argv)
        if argv[:2] == ["image", "inspect"] and "--format" not in argv:
            return json.dumps([{"RepoDigests": ["registry/image@sha256:" + "a" * 64]}])
        if argv[:2] == ["image", "inspect"]:
            return "sha256:" + "b" * 64
        if argv[0] == "run":
            return json.dumps({"pip_constraints": ["torch==2.12.0"]})
        if argv[0] == "create":
            return argv[argv.index("--name") + 1] + "-id"
        return ""

    monkeypatch.setattr(backend, "run", run)
    backend.build()
    for group in plan["groups"]:
        backend.evaluate_group(
            group,
            "benchmark_datasets/gsm8k/test.jsonl",
            "benchmark_datasets/sharegpt.json",
        )
    assert len([x for x in calls if x[:2] == ["docker", "build"]]) == 1
    creations = [x for x in calls if x[0] == "create"]
    assert len(creations) == 3
    assert len({x[x.index("--name") + 1] for x in creations}) == 3
    executions = [x for x in calls if x[:2] == ["docker", "exec"]]
    assert len(executions) == 3  # Each group driver executes its pair sequentially.
    assert len([x for x in calls if x[:2] == ["rm", "-f"]]) == 3
    assert not backend.containers
    assert all("--privileged" not in x for x in creations)


def test_failed_group_still_removes_its_container(tmp_path, monkeypatch):
    class Process:
        def __init__(self, argv, log_path):
            pass

        def wait(self, timeout, progress=False):
            return 1

    monkeypatch.setattr("docker_backend.ManagedProcess", Process)
    plan = make_plan(run_id="failed-group")
    backend = DockerBackend(plan, tmp_path)
    calls = []
    monkeypatch.setattr(
        backend, "run", lambda argv: calls.append(argv) or "container-id"
    )
    with pytest.raises(RuntimeError):
        backend.evaluate_group(plan["groups"][0], "gsm8k.jsonl", "sharegpt.json")
    assert calls[-1] == ["rm", "-f", "container-id"]
    assert not backend.containers


def test_creation_failure_does_not_claim_an_existing_container(tmp_path, monkeypatch):
    plan = make_plan(run_id="collision")
    backend = DockerBackend(plan, tmp_path)

    def create_fails(argv):
        assert argv[0] == "create"
        raise RuntimeError("name already in use")

    monkeypatch.setattr(backend, "run", create_fails)
    with pytest.raises(RuntimeError, match="already in use"):
        backend.evaluate_group(plan["groups"][0], "gsm8k.jsonl", "sharegpt.json")
    assert backend.containers == []


def test_wrong_retry_lock_is_rejected_before_any_docker_mutation(tmp_path, monkeypatch):
    plan = make_plan(run_id="retry")
    lock = tmp_path / "image-lock.json"
    write_json(
        lock, {"plan_hash": "wrong", "base_image_digest": "image@sha256:" + "a" * 64}
    )
    backend = DockerBackend(plan, tmp_path, image_lock=lock)
    monkeypatch.setattr(
        backend,
        "run",
        lambda argv: pytest.fail("Docker must not run with an invalid lock"),
    )
    with pytest.raises(ValueError, match="another run plan"):
        backend.build()
