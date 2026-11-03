"""CPU checks for counters and all five HTTP workload entry points."""

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from download_data import ensure_dataset
from record_acceptance import instrument, summarize


def output(index, *, tokens=8, verifies=2, accepted=6, proposed=30):
    return {
        "id": index,
        "text": "\\boxed{2}\n```python\ndef answer():\n    return 2\n```",
        "meta_info": {
            "completion_tokens": tokens,
            "spec_verify_ct": verifies,
            "spec_num_correct_drafts": accepted,
            "spec_num_proposed_drafts": proposed,
            "spec_accept_length": tokens / verifies,
            "finish_reason": {"type": "stop"},
        },
    }


class CounterTests(unittest.TestCase):
    def test_weighted_length_and_strict_rate(self):
        result = summarize(
            [
                output(0, tokens=8, verifies=1, accepted=7, proposed=15),
                output(1, tokens=12, verifies=3, accepted=9, proposed=45),
            ]
        )
        self.assertEqual(result["spec_accept_length_weighted"], 5)
        self.assertAlmostEqual(result["spec_accept_rate"], 16 / 60)

    def test_partial_counters_do_not_produce_a_rate(self):
        second = output(1)
        del second["meta_info"]["spec_num_proposed_drafts"]
        del second["meta_info"]["spec_verify_ct"]
        result = summarize([output(0), second])
        self.assertIsNone(result["spec_accept_rate"])
        self.assertIsNone(result["spec_accept_length_weighted"])
        self.assertEqual(result["requests_with_draft_counters"], 1)

    def test_invalid_counters_fail(self):
        for value in (-1, 1.5, True, "2"):
            with self.subTest(value=value):
                item = output(0)
                item["meta_info"]["spec_num_correct_drafts"] = value
                with self.assertRaises(RuntimeError):
                    summarize([item])
        with self.assertRaises(RuntimeError):
            summarize([output(0, accepted=31, proposed=30)])

    def test_missing_completion_count_fails(self):
        item = output(0)
        del item["meta_info"]["completion_tokens"]
        with self.assertRaises(RuntimeError):
            summarize([item])

    def test_wrong_cached_dataset_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gsm8k.jsonl"
            path.write_text('{"problem": "2+2", "answer": "4"}\n')
            with self.assertRaises(RuntimeError):
                ensure_dataset("gsm8k", path)


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        owner = self
        self.algorithm = "DFLASH"
        self.omit_proposed = False

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, data):
                encoded = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                if self.path == "/server_info":
                    self.reply(
                        {
                            "speculative_algorithm": owner.algorithm,
                            "tp_size": 8,
                            "version": "fixture",
                        }
                    )
                elif self.path == "/flush_cache":
                    self.reply({})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                text = body["text"]

                def generate(prompt):
                    match = re.search(r"id=(\d+)", prompt)
                    index = int(match[1]) if match else -1
                    item = output(index)
                    if owner.omit_proposed:
                        del item["meta_info"]["spec_num_proposed_drafts"]
                    return item

                self.reply(
                    [generate(prompt) for prompt in text]
                    if isinstance(text, list)
                    else generate(text)
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def run_workload(self, workload, directory, *, batch=False):
        path = Path(__file__).parent / f"bench_dflash_{workload}_sweep.py"
        spec = importlib.util.spec_from_file_location(f"dflash_test_{workload}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: messages[0]["content"]
        )
        mod.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args: tokenizer)
        instrument(mod, directory / "acceptance")
        rows = []
        for index in range(7):
            marker = f"id={index}"
            rows.append(
                {
                    "question": marker,
                    "problem": marker,
                    "answer": "2",
                    "prompt": f"# {marker}\ndef answer():\n    pass\n",
                    "test": "def check(fn):\n    assert fn() == 2",
                    "entry_point": "answer",
                    "text": marker,
                    "test_list": ["assert answer() == 2"],
                    "turns": [marker],
                }
            )
        dataset = directory / "data.jsonl"
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
        argv = [
            str(path),
            "--base-url",
            self.base_url,
            "--external-mode",
            "dflash",
            "--target-model",
            "fixture",
            "--data-path",
            str(dataset),
            "--concurrencies",
            "2",
            "--questions-per-concurrency-base",
            "7",
            "--max-questions-per-config",
            "7",
            "--tp-sizes",
            "8",
            "--attention-backends",
            "aiter",
            "--result-json",
            str(directory / "metrics.json"),
        ]
        if batch:
            argv.append("--batch-requests")
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            mod.main()
        return json.loads((directory / "acceptance/acceptance.json").read_text())[0]

    def test_all_workloads_exclude_warmup_and_save_complete_counters(self):
        for workload in ("gsm8k", "math500", "humaneval", "mbpp", "mtbench"):
            for batch in (False, True):
                with (
                    self.subTest(workload=workload, batch=batch),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    directory = Path(temporary)
                    result = self.run_workload(workload, directory, batch=batch)
                    self.assertEqual(result["requests"], 5)
                    self.assertEqual(result["warmup_requests_excluded"], 2)
                    self.assertEqual(result["output_tokens"], 40)
                    self.assertEqual(result["spec_verify_ct_sum"], 10)
                    self.assertAlmostEqual(result["spec_accept_rate"], 0.2)
                    self.assertEqual(result["spec_accept_length_weighted"], 4)
                    self.assertEqual(result["server_config"]["tp_size"], 8)
                    self.assertEqual(
                        result["accuracy"], None if workload == "mtbench" else 1.0
                    )
                    recorded = [
                        json.loads(line)
                        for line in (directory / "acceptance/run1_responses.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    self.assertEqual({item["id"] for item in recorded}, {2, 3, 4, 5, 6})

    def test_incomplete_counters_fail_without_publishing_summary(self):
        self.omit_proposed = True
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "Incomplete DFlash"):
                self.run_workload("math500", directory)
            self.assertFalse((directory / "acceptance/acceptance.json").exists())

    def test_eagle_server_cannot_be_reported_as_dflash(self):
        self.algorithm = "EAGLE"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "not configured for DFLASH"):
                self.run_workload("math500", Path(temporary))


if __name__ == "__main__":
    unittest.main()
