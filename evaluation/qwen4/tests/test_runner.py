import json
import socket
import sys

import runner
from config import make_plan
from process import port_available


def test_client_failure_keeps_other_evaluation_and_cleans_server(tmp_path, monkeypatch):
    server = tmp_path / "server.py"
    server.write_text("""import http.server,json,sys
info=json.loads(sys.argv[2])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        data={"data":[{"id":info["served_model_name"]}]} if self.path=="/v1/models" else info
        self.send_response(200);self.end_headers();self.wfile.write(json.dumps(data).encode())
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200);self.end_headers()
        self.wfile.write(b'{"choices":[{"message":{"content":"2"}}]}')
    def log_message(self,*args): pass
http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
""")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    plan = make_plan(profile="smoke", selection=["q35-35b-tp4-base"], port=port)
    group, case = plan["groups"][0], plan["groups"][0]["cases"][0]

    def launch(case, group, port, models_root, python):
        info = {
            "served_model_name": case["served_model_name"],
            "model_path": f"{models_root}/{group['directory']}",
            "tp_size": case["tp"],
            "speculative_algorithm": None,
            "disable_radix_cache": True,
        }
        return [sys.executable, str(server), str(port), json.dumps(info)]

    monkeypatch.setattr(runner, "server_command", launch)
    monkeypatch.setattr(
        runner,
        "perf_command",
        lambda *args: [sys.executable, "-c", "raise SystemExit(4)"],
    )
    monkeypatch.setattr(
        runner,
        "accuracy_command",
        lambda *args: [sys.executable, "-c", "print('accuracy still ran')"],
    )
    state = runner.run_case(
        plan,
        group,
        case,
        tmp_path / "output",
        "/models",
        "/unused-gsm8k",
        "/unused-sharegpt",
    )
    assert state["status"] == "eval_error"
    assert state["repeats"][0]["performance"] == 4
    assert state["repeats"][0]["accuracy"] == 0
    assert state["repeats"][0]["startup_s"] >= 0
    assert state["repeats"][0]["cleanup_s"] >= 0
    assert state["current_phase"] == "finished"
    assert port_available(port)
