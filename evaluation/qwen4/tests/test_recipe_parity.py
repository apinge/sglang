"""Compare generated commands with executable snapshots of the reviewed recipes.

The stubs record argv/environment; they never import torch or start a server.
Snapshots came from SGLang 91cdd01's evaluation/ directory.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from config import make_plan, server_command, server_environment


def options(argv):
    result = {}
    i = 0
    while i < len(argv):
        key = argv[i]
        if not key.startswith("--"):
            i += 1
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            result[key] = argv[i + 1].rstrip("/")
            i += 2
        else:
            result[key] = True
            i += 1
    for key in ["--host", "--port", "--served-model-name"]:
        result.pop(key, None)  # Deliberately owned by the CI supervisor.
    return result


@pytest.mark.parametrize(
    "index,filename",
    [
        (0, "launch_qwen3.5-35B-fp8_tp4_disable_prefix_cache.sh"),
        (1, "launch_qwen3.5-35B-fp8_tp1_dflash_with_prefix_cache_mfs0.7_bs32.sh"),
        (2, "launch_qwen3.5-397B-fp8_tp8_disable_prefix_cache.sh"),
        (3, "launch_qwen3.5-397B-fp8_tp8_dflash_with_prefix_cache.sh"),
        (4, "launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh"),
        (5, "launch_qwen38_flash_next_fp8_mi308x_pure_tp_4_or_8_or_2.sh"),
    ],
)
def test_recipe_matches_reviewed_shell(tmp_path, index, filename):
    plan = make_plan()
    group, case = [(g, c) for g in plan["groups"] for c in g["cases"]][index]
    bins = tmp_path / "bin"
    bins.mkdir()
    stub = (
        f"#!{sys.executable}\n"
        + """import json, os, sys
if sys.argv[1:2] != ['-']:
    selected = {k:v for k,v in os.environ.items() if k.startswith(('SGLANG_', 'AITER_', 'HIP_GDN_', 'USE_AITER_', 'TVM_FFI_'))}
    with open(os.environ['CAPTURE'], 'w') as f:
        json.dump({'argv': sys.argv[1:], 'env': selected}, f)
"""
    )
    for name in ["python", "python3", "sglang"]:
        (bins / name).write_text(stub)
        (bins / name).chmod(0o755)
    recipe = tmp_path / filename
    shutil.copyfile(Path(__file__).parent / "recipes" / filename, recipe)
    model = tmp_path / group["directory"]
    model.mkdir()
    (model / "config.json").write_text("{}")
    capture = tmp_path / "capture.json"
    env = {
        "PATH": f"{bins}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "MODEL_PATH": str(model),
    }
    subprocess.run(
        ["bash", str(recipe)], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    expected = json.loads(capture.read_text())
    new_env = {
        **env,
        **server_environment(
            case, group, models_root=str(tmp_path), python=str(bins / "python")
        ),
        "LOG_FILE": str(tmp_path / "new-launch.log"),
    }
    subprocess.run(
        server_command(case, group),
        cwd=tmp_path,
        env=new_env,
        check=True,
        capture_output=True,
    )
    actual = json.loads(capture.read_text())
    expected_args, actual_args = options(expected["argv"]), options(actual["argv"])
    if index == 4:
        expected_args = {
            key: value
            for key, value in expected_args.items()
            if not key.startswith("--speculative-")
        }
        assert not any(key.startswith("--speculative-") for key in actual_args)
    for key in ["--model-path", "--speculative-draft-model-path"]:
        if key in expected_args:
            assert Path(expected_args[key]).name == Path(actual_args[key]).name
            expected_args[key] = actual_args[key]
    assert expected_args == actual_args
    assert expected["env"] == actual["env"]
