"""Run native Qwen4 CI with the selected prepared AITER tables in server environments."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def server_environment(original, configs_dir, phase):
    configs_dir = Path(configs_dir).resolve()

    def configured(**extra):
        env = original(**extra)
        if "MODEL_PATH" in extra:
            env.update(
                SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB="1",
                SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE="1",
                AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=str(
                    configs_dir / ("ck_" + phase + ".csv")
                ),
                AITER_CONFIG_GEMM_BF16=str(configs_dir / ("tgemm_" + phase + ".csv")),
            )
        return env

    return configured


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--configs-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("base", "accepted"), required=True)
    args, native_args = parser.parse_known_args()
    for family in ("ck", "tgemm"):
        if not (args.configs_dir / (family + "_" + args.phase + ".csv")).is_file():
            raise FileNotFoundError("Prepare complete configuration tables before CI")
    sys.path.insert(0, str(ROOT / "scripts/ci"))
    import qwen4_ci as ci

    ci.clean_environment = server_environment(
        ci.clean_environment, args.configs_dir, args.phase
    )
    sys.argv = [sys.argv[0], *native_args]
    return ci.main()


if __name__ == "__main__":
    raise SystemExit(main())
