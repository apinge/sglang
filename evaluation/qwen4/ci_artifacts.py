"""Select the latest completed artifact when only a failed report job is retried."""

import argparse
import os
import re
import sys


def latest_result(names, attempt):
    candidates = []
    for name in names:
        match = re.fullmatch(r"qwen4-results-(\d+)", name.strip())
        if match and int(match[1]) <= attempt:
            candidates.append((int(match[1]), name.strip()))
    return max(candidates)[1] if candidates else ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--find", help="Require restoration when this named artifact exists"
    )
    args = parser.parse_args()
    names = sys.stdin.read().splitlines()
    name = (
        (args.find if args.find in names else "")
        if args.find
        else latest_result(names, int(os.environ["GITHUB_RUN_ATTEMPT"]))
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"name={name}\n")
