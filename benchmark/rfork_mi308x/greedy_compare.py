import json
import sys
import urllib.request

prompts = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
    "def fibonacci(n):",
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nA:",
    "Translate to Chinese: Machine learning models need large amounts of data.",
    "List three prime numbers greater than 100:",
    "Once upon a time, in a small village by the sea,",
    "The derivative of x^3 * sin(x) is",
]


def gen(port, p):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(
            {"text": p, "sampling_params": {"temperature": 0, "max_new_tokens": 128}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=300))["text"]


same = 0
for p in prompts:
    a, b = gen(int(sys.argv[1]), p), gen(int(sys.argv[2]), p)
    same += a == b
    print("SAME" if a == b else "DIFF", repr(p[:40]))
    if a != b:
        i = next(
            (k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b))
        )
        print(
            "   first diff at char",
            i,
            "| seed:",
            repr(a[max(0, i - 20) : i + 40]),
            "| client:",
            repr(b[max(0, i - 20) : i + 40]),
        )
print(f"{same}/{len(prompts)} identical")
