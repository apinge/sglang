"""Optional installed-runtime integration: real pinned scorer, fake HTTP server."""

import argparse
import http.server
import json
import re
import threading

import pytest
from common import read_json
from results import validate_accuracy

from assets import gsm8k_selection


@pytest.mark.parametrize(
    "http_error,truncate", [(False, False), (True, False), (False, True)]
)
def test_audited_sampler_counts_errors_without_changing_scorer(
    tmp_path, http_error, truncate
):
    pytest.importorskip("sglang", reason="Installed-runtime client integration")
    from sglang.test.run_eval import run_eval

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if http_error:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"test failure"}}')
                return
            index = int(
                re.findall(r"Question: Q(\d+)", body["messages"][0]["content"])[-1]
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "id": "fixture",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "" if truncate else str(index),
                                },
                                "finish_reason": "length" if truncate else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 1,
                            "total_tokens": 101,
                        },
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    data = tmp_path / "test.jsonl"
    data.write_text(
        "".join(
            json.dumps({"question": f"Q{i}", "answer": f"#### {i}"}) + "\n"
            for i in range(7)
        )
    )
    args = argparse.Namespace(
        eval_name="gsm8k",
        gsm8k_data_path=str(data),
        num_examples=2,
        num_shots=5,
        num_threads=2,
        output_dir=str(tmp_path / "results"),
        base_url=None,
        host="127.0.0.1",
        port=server.server_port,
        model="test-model",
        max_tokens=32,
        repeat=1,
        api="chat",
        strict=True,
        temperature=0.0,
        top_p=1.0,
    )
    try:
        if http_error:
            with pytest.raises(RuntimeError, match="Incomplete GSM8K"):
                run_eval(args)
        else:
            run_eval(args)
        result = read_json(tmp_path / "results/gsm8k.json")
        assert result["scored"] == 2
        if http_error:
            assert result["errors"] == 2 and result["completed"] == 0
            with pytest.raises(ValueError):
                validate_accuracy(
                    tmp_path / "results/gsm8k.json", 2, gsm8k_selection(data, 2)
                )
        else:
            assert result["score"] == (0 if truncate else 1)
            if truncate:
                assert result["empty_outputs"] == 2 and result["truncated"] == 2
            validate_accuracy(
                tmp_path / "results/gsm8k.json", 2, gsm8k_selection(data, 2)
            )
    finally:
        server.shutdown()
        server.server_close()
