"""Regression coverage for the accuracy gate and owned-process cleanup."""

import argparse
import contextlib
import http.server
import json
import os
import subprocess
import sys
import threading

import pytest
import qwen4_ci as ci


def accuracy_result(tmp_path, score=0.97, examples=200):
    path = tmp_path / "gsm8k.json"
    path.write_text(
        json.dumps(
            dict(
                score=score,
                requested=examples,
                scored=examples,
                completed=examples,
                errors=0,
            )
        )
    )
    path.with_suffix(".html").write_text("<html>results</html>")
    return path


@pytest.mark.parametrize(
    "case,minimum_count",
    zip(ci.CASES, (191, 192, 191, 189, 189, 193, 192)),
    ids=[case[0] for case in ci.CASES],
)
def test_accuracy_gate_preserves_each_baseline_and_rejects_the_next_lower_count(
    case, minimum_count
):
    baseline = case[5]
    ci.check_accuracy_gate(minimum_count / 200, baseline)
    with pytest.raises(ValueError, match="regression"):
        ci.check_accuracy_gate((minimum_count - 1) / 200, baseline)
    ci.check_accuracy_gate(1, baseline)


@pytest.mark.parametrize(
    "key,value",
    [
        ("requested", 199),
        ("scored", 199),
        ("completed", 199),
        ("errors", 1),
        ("score", float("nan")),
        ("score", float("inf")),
        ("score", True),
        ("score", -0.01),
        ("score", 1.01),
        ("score", 0.971),
        ("errors", False),
    ],
)
def test_incomplete_or_invalid_accuracy_cannot_pass(tmp_path, key, value):
    path = accuracy_result(tmp_path)
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ci.validate_accuracy(path)


def test_smoke_accuracy_is_diagnostic_but_must_be_complete(tmp_path):
    path = accuracy_result(tmp_path, score=0, examples=2)
    ci.validate_accuracy(path, smoke=True)
    ci.check_accuracy_gate(0, 0.97, smoke=True)
    with pytest.raises(ValueError, match="200"):
        ci.validate_accuracy(path)


def test_unlimited_benchmark_metadata_does_not_break_summary_json(tmp_path):
    data = dict(
        completed=2,
        errors=["", ""],
        output_lens=[500, 500],
        mean_ttft_ms=100,
        mean_tpot_ms=2,
        output_throughput=100,
        request_throughput=0.2,
        accept_length=4,
        request_rate=float("inf"),
    )
    path = tmp_path / "performance.jsonl"
    path.write_text(json.dumps(data) + "\n")
    metrics = ci.validate_performance(path, 2, 500, speculative=True)
    ci.write_json(tmp_path / "summary.json", metrics)
    assert metrics["output_throughput"] == 100 and "request_rate" not in metrics


def test_fresh_results_required_and_unknown_cases_rejected(tmp_path):
    for argv, message in [
        (["--case", "unknown"], "Unknown cases"),
        ([], "fresh output"),
    ]:
        (tmp_path / "old.json").write_text("{}")
        result = subprocess.run(
            [
                sys.executable,
                str(ci.ROOT / "scripts/ci/qwen4_ci.py"),
                "--output",
                str(tmp_path),
                *argv,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2 and message in result.stderr


def test_accuracy_regression_fails_the_run_and_reports_remaining_cases(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ci,
        "command",
        lambda argv, **kwargs: ci.CLIENT_SHA if argv[-1] == "HEAD" else "",
    )
    monkeypatch.setattr(ci, "DATASETS", {})
    visited = []

    def case_run(case, args, directory, row):
        visited.append(case[0])
        row["accuracy"] = ci.validate_accuracy(accuracy_result(directory, score=0.9))
        ci.check_accuracy_gate(row["accuracy"]["score"], case[5])

    monkeypatch.setattr(ci, "run_case", case_run)
    args = argparse.Namespace(
        case=",".join(case[0] for case in ci.CASES[:2]),
        output=tmp_path / "results",
        client_root=tmp_path,
        models_root=tmp_path,
        datasets_root=None,
        accuracy_only=True,
        lock=tmp_path / "gpu.lock",
        smoke=False,
    )
    assert ci.run(args) == 1
    assert visited == [case[0] for case in ci.CASES[:2]]
    report = json.loads((args.output / "summary.json").read_text())
    assert report["failed"] and all(
        row["status"] == "failed" for row in report["cases"]
    )
    assert "90.00%" in (args.output / "summary.md").read_text()


@pytest.mark.parametrize(
    "case_id", ["Qwen3.8-Flash-TP2-Base", "Qwen3.8-Flash-TP2-RFork"]
)
@pytest.mark.parametrize("pressure", [False, True])
@pytest.mark.parametrize(
    "execution_attempt,conclusion,has_report,has_multimodal,has_rocjpeg",
    [
        (2, "success", True, True, True),
        (6, "success", True, True, True),
        (6, "failure", True, True, True),
        (6, "failure", False, True, True),
        (6, "success", True, False, True),
        (6, "success", True, True, False),
    ],
)
def test_summary_uses_actual_execution_and_keeps_latest_conclusion(
    tmp_path,
    monkeypatch,
    case_id,
    pressure,
    execution_attempt,
    conclusion,
    has_report,
    has_multimodal,
    has_rocjpeg,
):
    definition = next(case for case in ci.CASES if case[0] == case_id)
    case, baseline = definition[0], definition[5]
    is_rfork = definition[3] == "rfork"
    original = {
        "name": f"Qwen4 ({case})",
        "run_attempt": 2,
        "started_at": "2026-11-04T07:26:19Z",
        "conclusion": "success",
    }
    execution = dict(original, run_attempt=execution_attempt, conclusion=conclusion)
    if execution_attempt == 6:
        execution["started_at"] = "2026-11-05T11:57:29Z"
    copied = dict(execution, run_attempt=7)
    # Return history out of order and across pages, as the API may do.
    pages = [{"jobs": [copied]}, {"jobs": [execution, original]}]
    monkeypatch.setattr(ci, "command", lambda *args, **kwargs: json.dumps(pages))
    for key, value in {
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_REPOSITORY": "apinge/sglang",
        "GITHUB_RUN_ID": "41785211796",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary.md"),
        "CASE_MATRIX": json.dumps({"case": [case]}),
        "VALIDATION_RESULT": "success",
        "CASES_RESULT": "success",
        "DAILY_PRESSURE": "true" if pressure else "false",
    }.items():
        monkeypatch.setenv(key, value)
    for attempt, score in [(2, 0.98)] + (
        [(6, 0.99)] if execution_attempt == 6 and has_report else []
    ):
        directory = (
            tmp_path
            / "qwen4-case-results"
            / f"qwen4-results-{case}-{attempt}"
            / "evaluation"
        )
        directory.mkdir(parents=True)
        row = dict(
            case=case,
            baseline=baseline,
            status="passed",
            errors=[],
            accuracy={"score": score},
        )
        if is_rfork:
            row["rfork"] = {
                "startup_s": 71.1,
                "load_weight_s": 13.02,
                "tensors_per_rank": 1504,
            }
        elif has_multimodal:
            row["multimodal"] = {"status": "passed", "completion_tokens": 10}
            if has_rocjpeg:
                row["multimodal"]["rocjpeg"] = "on"
        ci.summary(
            directory, [row], argparse.Namespace(smoke=False, pressure=pressure), []
        )
    passed = conclusion == "success" and (is_rfork or (has_multimodal and has_rocjpeg))
    assert ci.summarize_ci() == (not passed)
    row = json.loads((tmp_path / "qwen4-summary/summary.json").read_text())["cases"][0]
    assert row["status"] == ("passed" if passed else "failed")
    if has_report:
        assert row["accuracy"]["score"] == (0.98 if execution_attempt == 2 else 0.99)
    else:
        assert "accuracy" not in row
        assert "qwen4-results-" + case + "-6" in row["errors"][0]
    if has_report and not is_rfork and not has_multimodal:
        assert "Missing or failed multimodal result" in row["errors"]
    if has_report and not is_rfork and has_multimodal and not has_rocjpeg:
        assert "Missing rocJPEG activation confirmation" in row["errors"]
    rendered = (tmp_path / "qwen4-summary/summary.md").read_text()
    assert f"Coverage: 1/{len(ci.CASES)}" in rendered
    if is_rfork:
        assert "not applicable" in rendered
        if has_report:
            assert "1504 tensors per rank match the seed" in rendered


def multimodal_response(model="test-model", content="The image shows a triangle."):
    return {
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "completion_tokens": 8,
            "prompt_tokens_details": {"image_tokens": 156},
        },
    }


@pytest.mark.parametrize("correction", ["on", "off"])
def test_rocjpeg_success_metric_is_not_a_server_fault(tmp_path, correction):
    path = tmp_path / "server.log"
    message = (
        f"[server] rocJPEG GPU JPEG decoding: on, color correction {correction} "
        "(mean abs error vs PIL 0.58)"
    )
    path.write_text(message + "\nrocJPEG request batch decoded: 1 image(s)\n")
    ci.check_rocjpeg_log(path)
    ci.check_server_log(path)
    path.write_text(message + "; GPU fault detected\n")
    with pytest.raises(RuntimeError, match="GPU fault"):
        ci.check_server_log(path)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "empty",
        "reasoning_only",
        "wrong_model",
        "error",
        "no_tokens",
        "no_image_tokens",
        "bad_choices",
    ],
)
def test_multimodal_missing_or_invalid_response_fails(tmp_path, failure):
    path = tmp_path / "multimodal.json"
    data = multimodal_response()
    if failure == "empty":
        data["choices"][0]["message"]["content"] = " "
    elif failure == "reasoning_only":
        data["choices"][0]["message"].update(content=None, reasoning_content="thinking")
    elif failure == "wrong_model":
        data["model"] = "another-model"
    elif failure == "error":
        data = {"error": {"message": "decode failed"}}
    elif failure == "no_tokens":
        data["usage"]["completion_tokens"] = 0
    elif failure == "no_image_tokens":
        data["usage"]["prompt_tokens_details"]["image_tokens"] = 0
    elif failure == "bad_choices":
        data["choices"] = []
    if failure != "missing":
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        ci.validate_multimodal(path, "test-model")


@pytest.mark.parametrize("status", [200, 500])
def test_multimodal_script_targets_case_and_propagates_http_errors(tmp_path, status):
    seen = []
    model = 'candidate/"quoted"'
    data = (
        multimodal_response(model)
        if status == 200
        else {"error": {"message": "image decoding failed"}}
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(
                (
                    self.path,
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, *args):
            pass

    output = tmp_path / "multimodal.json"
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                ["bash", str(ci.EVALUATION / "run_multimodal_test.sh")],
                env={
                    **os.environ,
                    "CLIENT_PYTHON": sys.executable,
                    "BENCH_HOST": "127.0.0.1",
                    "BENCH_PORT": str(server.server_port),
                    "BENCH_SERVED_MODEL": model,
                    "MM_IMAGE_URL": "https://example.com/image.jpg",
                    "MM_OUTPUT_FILE": str(output),
                },
                text=True,
                capture_output=True,
                timeout=15,
            )
        finally:
            server.shutdown()
            thread.join()
    assert (result.returncode == 0) == (status == 200)
    assert json.loads(output.read_text()) == data
    endpoint, body = seen[0]
    assert endpoint == "/v1/chat/completions" and body["model"] == model
    assert (
        body["messages"][0]["content"][0]["image_url"]["url"]
        == "https://example.com/image.jpg"
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    if status == 200:
        assert ci.validate_multimodal(output, model)["status"] == "passed"


@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize(
    "rocjpeg", ["on", "off", "missing", "self_check_only", "batch_fallback"]
)
def test_case_runs_multimodal_after_text_phases_and_gates_failure(
    tmp_path, monkeypatch, valid, rocjpeg
):
    case = ci.CASES[0]
    model = tmp_path / case[1]
    model.mkdir()
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(ci, "select_gpus", lambda *args: ["0", "1"])

    @contextlib.contextmanager
    def managed_process(argv, log, env, cwd):
        yield object()
        # Only available after the log relay drains, as in the real lifecycle.
        if rocjpeg != "missing":
            activation = "off" if rocjpeg == "off" else "on"
            message = f"[server] rocJPEG GPU JPEG decoding: {activation}, color correction on\n"
            if rocjpeg in ("on", "batch_fallback"):
                message += "rocJPEG request batch decoded: 1 image(s)\n"
            if rocjpeg == "batch_fallback":
                # An earlier successful batch must not hide a later fallback.
                message += "rocJPEG batch decode failed, decoding with PIL: injected\n"
            log.write_text(message)

    monkeypatch.setattr(ci, "managed_process", managed_process)
    monkeypatch.setattr(ci, "wait_ready", lambda *args: {})
    phases = []

    def run_phase(script, directory, env, server, timeout):
        phases.append(script)
        if "pure_text" in script:
            (directory / "performance.jsonl").write_text(
                json.dumps(
                    dict(
                        completed=2,
                        errors=["", ""],
                        output_lens=[350, 350],
                        mean_ttft_ms=100,
                        mean_tpot_ms=2,
                        output_throughput=100,
                        request_throughput=1,
                    )
                )
                + "\n"
            )
        elif script == "run_gsm8k.sh":
            accuracy_result(directory, score=1, examples=2)
        else:
            ci.write_json(
                env["MM_OUTPUT_FILE"],
                multimodal_response(
                    env["BENCH_SERVED_MODEL"], "A triangle" if valid else ""
                ),
            )

    monkeypatch.setattr(ci, "run_phase", run_phase)
    args = argparse.Namespace(
        models_root=tmp_path,
        gpus="0,1",
        port=0,
        server_timeout=1,
        datasets_root=None,
        client_root=tmp_path,
        smoke=True,
        accuracy_only=False,
    )
    row = {"errors": []}
    ci.run_case(case, args, tmp_path, row)
    assert phases == [
        "run_qwen38_pure_text_bench.sh",
        "run_gsm8k.sh",
        "run_multimodal_test.sh",
    ]
    passed = valid and rocjpeg == "on"
    assert row["multimodal"]["status"] == ("passed" if passed else "failed")
    assert bool(row["errors"]) == (not passed)
    if passed:
        assert row["multimodal"]["rocjpeg"] == "on"
    elif valid and rocjpeg == "off":
        assert "rocJPEG GPU decoding was not confirmed" in row["errors"][0]
    elif valid and rocjpeg == "self_check_only":
        assert "request batch decoding was not confirmed" in row["errors"][0]
    elif valid and rocjpeg == "batch_fallback":
        assert "request batch fell back to PIL" in row["errors"][0]
    assert (
        json.loads((tmp_path / "launch.json").read_text())["env"][
            "SGLANG_ROCM_GPU_JPEG_DECODE"
        ]
        == "1"
    )


def test_gpu_selection_skips_busy_devices_and_maps_to_host_indices(monkeypatch):
    seen = {}

    def run(argv, env, **kwargs):
        seen["visible"] = env["HIP_VISIBLE_DEVICES"]
        return subprocess.CompletedProcess(argv, 0, stdout="[1, 3]\n", stderr="")

    monkeypatch.setattr(ci.subprocess, "run", run)
    assert ci.select_gpus({}, ["0", "2", "5", "7"], 2) == ["2", "7"]
    assert seen["visible"] == "0,2,5,7"
    assert ci.select_gpus({}, ["0", "2", "5", "7"], 1) == ["2"]
    with pytest.raises(RuntimeError, match="Need 4 idle GPUs.*idle: 2,7"):
        ci.select_gpus({}, ["0", "2", "5", "7"], 4)


def write_checksums(directory, rows):
    directory.mkdir()
    for rank, lines in rows.items():
        (directory / f"target_rank{rank}.tsv").write_text(
            "".join("\t".join(line) + "\n" for line in lines)
        )


def test_rfork_weight_checksums_must_match_on_every_rank(tmp_path):
    seed = [
        ("param", "w", "(2, 2)", "torch.bfloat16", "cuda", "8:10:30"),
        ("buffer", "b", "(2,)", "torch.float32", "cuda", "8:4:9"),
    ]
    write_checksums(tmp_path / "seed", {0: seed, 1: seed})
    # The device column is not compared.
    same = [line[:4] + ("cpu",) + line[5:] for line in seed]
    write_checksums(tmp_path / "same", {0: same, 1: same})
    assert ci.compare_weight_checksums(tmp_path / "seed", tmp_path / "same", 2) == 2
    changed = [seed[0][:5] + ("8:10:31",), seed[1]]
    write_checksums(tmp_path / "changed", {0: seed, 1: changed})
    with pytest.raises(ValueError, match="rank 1: 1/2 differ"):
        ci.compare_weight_checksums(tmp_path / "seed", tmp_path / "changed", 2)
    write_checksums(tmp_path / "missing", {0: seed, 1: seed[:1]})
    with pytest.raises(ValueError, match="rank 1: 1/2 differ"):
        ci.compare_weight_checksums(tmp_path / "seed", tmp_path / "missing", 2)


def test_rfork_case_needs_a_seed_and_a_fork_worth_of_gpus(tmp_path):
    case = next(case for case in ci.CASES if case[3] == "rfork")
    (tmp_path / case[1]).mkdir()
    (tmp_path / case[1] / "config.json").write_text("{}")
    args = argparse.Namespace(models_root=tmp_path, gpus="0,1,2", port=7080)
    with pytest.raises(ValueError, match=f"requires {2 * case[2]} GPUs"):
        ci.run_case(case, args, tmp_path, {"errors": []})


def test_daily_only_cases_run_on_schedule_or_when_selected():
    ids = [case[0] for case in ci.CASES]
    daily = ci.DAILY_ONLY_CASES
    assert daily <= set(ids)
    assert ci.matrix_cases(set(), "pull_request") == [i for i in ids if i not in daily]
    assert ci.matrix_cases(set(), "schedule") == ids
    assert ci.matrix_cases(set(daily), "pull_request") == [i for i in ids if i in daily]
    with pytest.raises(ValueError, match="Unknown cases"):
        ci.matrix_cases({"bogus"}, "schedule")
