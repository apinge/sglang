"""Standard-library tests for validated-overlay merges; no ROCm runtime imports."""

import csv
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
KIT = ROOT / "benchmark/rocm_tgemm"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The registry itself is stdlib-only; avoid importing the runtime-heavy public API.
registry = load_file(
    "_rocm_tgemm_config_registry", ROOT / "python/sglang/test/ci/ci_register.py"
)
register_cpu_ci = registry.register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")
prepare = load_file("_rocm_tgemm_prepare_configs", KIT / "prepare_configs.py")
ci_wrapper = load_file("_rocm_tgemm_run_ci", KIT / "run_ci.py")


class TestRocmTgemmConfigs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_table(self, name, names, rows):
        path = self.directory / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def base_tables(self):
        paths = {}
        for family, path in prepare.OVERLAYS.items():
            names, rows = prepare.read_table(path, family)
            matched = dict(rows[0], libtype="old_backend")
            control = dict(rows[0], M="1", N="4096", K="2048")
            paths[family] = self.write_table(
                family + "_base.csv", names, [matched, control]
            )
        return paths

    def test_validated_rows_have_recorded_scope(self):
        self.assertEqual(len(prepare.read_table(prepare.OVERLAYS["ck"], "ck")[1]), 1)
        self.assertEqual(
            len(prepare.read_table(prepare.OVERLAYS["tgemm"], "tgemm")[1]), 4
        )
        for family, path in prepare.OVERLAYS.items():
            for row in prepare.read_table(path, family)[1]:
                self.assertEqual(row["gfx"], "gfx942")
                self.assertEqual(int(row["cu_num"]), 80)

    def test_merge_preserves_unmatched_rows_and_replaces_exact_key(self):
        for family, base_path in self.base_tables().items():
            with self.subTest(family=family):
                names, base = prepare.read_table(base_path, family)
                overlay_names, overlay = prepare.read_table(
                    prepare.OVERLAYS[family], family
                )
                output_names, output = prepare.merge_tables(
                    names, base, overlay_names, overlay, family
                )
                self.assertEqual(output_names, names)
                self.assertEqual(output[0], base[1])
                self.assertEqual(output[1:], overlay)
                self.assertEqual(len(output), len(overlay) + 1)

    def test_tgemm_key_retains_dtype_bias_scale_and_packing(self):
        names, overlay = prepare.read_table(prepare.OVERLAYS["tgemm"], "tgemm")
        row = overlay[0]
        controls = [
            dict(row, **{field: value})
            for field, value in [
                ("dtype", "torch.bfloat16"),
                ("outdtype", "torch.float16"),
                ("bias", "True"),
                ("scaleAB", "False"),
                ("bpreshuffle", "False"),
            ]
        ]
        _, merged = prepare.merge_tables(names, [row, *controls], names, [row], "tgemm")
        self.assertEqual(merged, [*controls, row])

    def test_invalid_schema_and_row_width_fail(self):
        path = self.directory / "invalid.csv"
        path.write_text("M,N,K\n1,2,3\n")
        with self.assertRaisesRegex(ValueError, "schema"):
            prepare.read_table(path, "ck")
        names, rows = prepare.read_table(prepare.OVERLAYS["ck"], "ck")
        path = self.write_table("bad_width.csv", names, rows)
        with path.open("a") as stream:
            stream.write("1,2\n")
        with self.assertRaisesRegex(ValueError, "malformed"):
            prepare.read_table(path, "ck")

    def test_duplicate_numeric_shape_is_rejected(self):
        for family, overlay_path in prepare.OVERLAYS.items():
            with self.subTest(family=family):
                names, rows = prepare.read_table(overlay_path, family)
                duplicate = dict(rows[0], cu_num="80.0", libtype="different_backend")
                path = self.write_table(
                    family + "_duplicate.csv", names, [rows[0], duplicate]
                )
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    prepare.read_table(path, family)

    def test_prepare_preserves_inputs_and_complete_base_snapshots(self):
        bases = self.base_tables()
        original = {family: path.read_bytes() for family, path in bases.items()}
        output = self.directory / "merged"
        manifest = prepare.prepare_configs(bases["ck"], bases["tgemm"], output)
        for family, path in bases.items():
            self.assertEqual(path.read_bytes(), original[family])
            self.assertEqual(
                (output / (family + "_base.csv")).read_bytes(), original[family]
            )
            self.assertEqual(
                manifest["tables"][family]["base_sha256"],
                hashlib.sha256(original[family]).hexdigest(),
            )
            _, merged = prepare.read_table(output / (family + "_accepted.csv"), family)
            count = 1 if family == "ck" else 4
            self.assertEqual(len(merged), count + 1)
            self.assertEqual(manifest["tables"][family]["overlay_rows"], count)
        self.assertTrue((output / "MANIFEST.json").is_file())

    def test_input_overwrite_is_rejected(self):
        bases = self.base_tables()
        original = bases["ck"].read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite an input"):
            prepare.prepare_configs(bases["ck"], bases["tgemm"], self.directory)
        self.assertEqual(bases["ck"].read_bytes(), original)

    def test_ci_server_environment_injects_without_changing_native_extra(self):
        calls = []

        def original(**extra):
            calls.append(dict(extra))
            return {"PATH": "/runtime/bin", **extra}

        configure = ci_wrapper.server_environment(original, self.directory, "accepted")
        extra = {
            "MODEL_PATH": "/models/model",
            "HIP_VISIBLE_DEVICES": "3,1",
            "PORT": "9000",
        }
        env = configure(**extra)
        self.assertEqual(calls, [extra])
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "3,1")
        self.assertEqual(env["PORT"], "9000")
        self.assertEqual(env["MODEL_PATH"], "/models/model")
        self.assertEqual(env["SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB"], "1")
        self.assertEqual(env["SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE"], "1")
        self.assertEqual(
            env["AITER_CONFIG_GEMM_BF16"], str(self.directory / "tgemm_accepted.csv")
        )
        self.assertEqual(
            env["AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE"],
            str(self.directory / "ck_accepted.csv"),
        )

    def test_ci_client_environment_is_unchanged(self):
        def original(**extra):
            return {"PATH": "/runtime/bin", **extra}

        configure = ci_wrapper.server_environment(original, self.directory, "base")
        extra = {
            "TOKENIZER_PATH": "/models/model",
            "HIP_VISIBLE_DEVICES": "6",
            "CUSTOM": "kept",
        }
        self.assertEqual(configure(**extra), original(**extra))

    def test_existing_measurement_configs_are_not_overwritten(self):
        bases = self.base_tables()
        output = self.directory / "merged"
        prepare.prepare_configs(bases["ck"], bases["tgemm"], output)
        original = (output / "ck_accepted.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exist"):
            prepare.prepare_configs(bases["ck"], bases["tgemm"], output)
        self.assertEqual((output / "ck_accepted.csv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
