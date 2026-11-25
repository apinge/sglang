"""End-to-end verification that --tokenizer-backend=fastokens swaps the
backend of the loaded tokenizer with fastokens' _TokenizerShim.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=14, suite="base-a-test-cpu")


try:
    import fastokens  # noqa: F401

    HAS_FASTOKENS = True
except ImportError:
    HAS_FASTOKENS = False


@unittest.skipUnless(HAS_FASTOKENS, "fastokens package not installed")
class TestFastokensBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import (
            Regex,
            Tokenizer,
            decoders,
            models,
            normalizers,
            pre_tokenizers,
            processors,
            trainers,
        )

        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.model = cls.directory.name
        cls.texts = [
            "किताब हिंदी महाराष्ट्र नमस्ते",
            "বাংলা ভাষা। हिन्दी किताब!",
            "Hello, world!\nकिताब 123",
            "Cafe\u0301 العربية नमस्ते",
        ]
        # Qwen3.5's checkpoint keeps combining marks in letter runs. The v5
        # Qwen2 constructor substitutes a regex without these marks.
        regex = (
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
            r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        )
        reference = Tokenizer(models.BPE())
        reference.normalizer = normalizers.NFC()
        reference.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(regex), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        reference.decoder = decoders.ByteLevel()
        reference.post_processor = processors.ByteLevel(trim_offsets=False)
        reference.train_from_iterator(
            cls.texts,
            trainer=trainers.BpeTrainer(
                vocab_size=512,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<|endoftext|>", "<|pad|>"],
                show_progress=False,
            ),
        )
        reference.save(str(Path(cls.model) / "tokenizer.json"))
        (Path(cls.model) / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "Qwen2Tokenizer",
                    "bos_token": "<|endoftext|>",
                    "eos_token": "<|endoftext|>",
                    "pad_token": "<|pad|>",
                    "chat_template": "{{ messages }}",
                }
            )
        )
        (Path(cls.model) / "config.json").write_text('{"model_type": "qwen2"}')
        cls.reference = reference

    def setUp(self):
        self._reset_fastokens()
        self.addCleanup(self._reset_fastokens)

    @staticmethod
    def _reset_fastokens():
        import fastokens

        from sglang.srt.utils.hf_transformers import tokenizer as tokenizer_utils

        fastokens.unpatch_transformers()
        tokenizer_utils._fastokens_patched = False

    def assert_checkpoint_ids(self, tokenizer):
        for text in self.texts:
            with self.subTest(text=text):
                self.assertEqual(
                    tokenizer.encode(text, add_special_tokens=False),
                    self.reference.encode(text, add_special_tokens=False).ids,
                )

    def test_shim_is_applied(self):
        # `_TokenizerShim` is fastokens' private compat shim. SGLang's
        # integration relies on `tokenizer._tokenizer` being an instance of
        # this class to confirm fastokens is wired up. If fastokens renames
        # or restructures it, update both this assertion and any code in
        # SGLang that depends on the same private name.
        from fastokens._compat import _TokenizerShim

        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            self.model,
            tokenizer_backend="fastokens",
        )
        backend = getattr(tokenizer, "_tokenizer", None)
        self.assertIsInstance(
            backend,
            _TokenizerShim,
            f"Expected tokenizer._tokenizer to be _TokenizerShim, "
            f"got {type(backend).__name__}",
        )

    def test_encode_decode_roundtrip(self):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            self.model,
            tokenizer_backend="fastokens",
        )
        text = "Hello, world!"
        ids = tokenizer.encode(text, add_special_tokens=False)
        self.assertGreater(len(ids), 0)
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), text)

    def test_tokenizer_preserves_checkpoint_regex(self):
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(self.model, tokenizer_backend="fastokens")
        self.assert_checkpoint_ids(tokenizer)

    def test_processor_preserves_checkpoint_regex(self):
        from transformers import AutoTokenizer

        from sglang.srt.utils.hf_transformers import processor as processor_utils

        for route in ("multimodal", "language_model_only", "deepseek_v41"):
            with self.subTest(route=route):
                config = SimpleNamespace(
                    model_type="deepseek_v41"
                    if route == "deepseek_v41"
                    else "test_vlm",
                    language_model_only=route == "language_model_only",
                    vision_n_layers=1,
                )
                with (
                    patch.object(
                        processor_utils,
                        "AutoConfig",
                        SimpleNamespace(from_pretrained=lambda *args, **kwargs: config),
                    ),
                    patch.object(
                        processor_utils.AutoProcessor,
                        "from_pretrained",
                        side_effect=lambda *args, **kwargs: SimpleNamespace(
                            tokenizer=AutoTokenizer.from_pretrained(*args, **kwargs)
                        ),
                    ) as auto_processor,
                ):
                    processor = processor_utils.get_processor(
                        self.model, tokenizer_backend="fastokens", local_files_only=True
                    )
                tokenizer = getattr(processor, "tokenizer", processor)
                self.assert_checkpoint_ids(tokenizer)
                self.assertEqual(auto_processor.call_count, int(route == "multimodal"))

    def test_huggingface_loaders_keep_existing_behavior(self):
        from transformers import AutoTokenizer

        from sglang.srt.utils.hf_transformers import processor as processor_utils
        from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

        original = AutoTokenizer.from_pretrained(self.model)
        expected = [
            original.encode(text, add_special_tokens=False) for text in self.texts
        ]
        tokenizer = get_tokenizer(self.model, tokenizer_backend="huggingface")
        with (
            patch.object(
                processor_utils,
                "AutoConfig",
                SimpleNamespace(
                    from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                        model_type="test_vlm"
                    )
                ),
            ),
            patch.object(
                processor_utils.AutoProcessor,
                "from_pretrained",
                side_effect=lambda *args, **kwargs: SimpleNamespace(
                    tokenizer=AutoTokenizer.from_pretrained(*args, **kwargs)
                ),
            ),
        ):
            processor = processor_utils.get_processor(
                self.model, tokenizer_backend="huggingface"
            )
        for loaded in (tokenizer, processor.tokenizer):
            self.assertEqual(type(loaded._tokenizer), type(original._tokenizer))
            self.assertEqual(
                [loaded.encode(text, add_special_tokens=False) for text in self.texts],
                expected,
            )

    def test_repair_preserves_live_state_and_is_idempotent(self):
        from tokenizers import processors
        from transformers import AutoTokenizer

        from sglang.srt.utils.hf_transformers import tokenizer as tokenizer_utils

        tokenizer_utils._ensure_fastokens_patched()
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        tokenizer.add_tokens(["<runtime-token>"])
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<runtime-special>"]}
        )
        backend = tokenizer._tokenizer
        tokenizer._tokenizer = backend = type(backend).from_str(
            backend.to_str(), pcre2_match_limit=123456
        )
        special_id = tokenizer.convert_tokens_to_ids("<runtime-special>")
        post_processor = processors.TemplateProcessing(
            single="<runtime-special> $A",
            special_tokens=[("<runtime-special>", special_id)],
        )
        backend.post_processor = post_processor
        backend.enable_truncation(max_length=10)
        backend.enable_padding(
            length=12, pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token
        )
        backend.encode_special_tokens = True
        vocabulary = backend.get_vocab()
        padding, truncation = backend.padding, backend.truncation

        tokenizer_utils._fix_fastokens_tokenizer_components(tokenizer, self.model)
        repaired = tokenizer._tokenizer
        self.assertEqual(repaired.get_vocab(), vocabulary)
        self.assertEqual(repaired.padding, padding)
        self.assertEqual(repaired.truncation, truncation)
        self.assertEqual(str(repaired.post_processor), str(backend.post_processor))
        self.assertTrue(repaired.encode_special_tokens)
        self.assertEqual(repaired._pcre2_options, backend._pcre2_options)
        for text in self.texts:
            encoded = repaired.encode(text)
            self.assertEqual(len(encoded.ids), 12)
            self.assertEqual(encoded.ids[0], special_id)
        tokenizer_utils._fix_fastokens_tokenizer_components(tokenizer, self.model)
        self.assertIs(tokenizer._tokenizer, repaired)

    def test_fastokens_uses_same_revision_for_loader_and_components(self):
        from sglang.srt.utils.hf_transformers import tokenizer as tokenizer_utils

        for kwargs, expected in (
            ({"tokenizer_revision": "checkpoint-revision"}, "checkpoint-revision"),
            (
                {"tokenizer_revision": "alias", "revision": "explicit-revision"},
                "explicit-revision",
            ),
        ):
            with (
                self.subTest(kwargs=kwargs),
                patch.object(
                    tokenizer_utils,
                    "_auto_tokenizer_from_pretrained",
                    wraps=tokenizer_utils._auto_tokenizer_from_pretrained,
                ) as load,
                patch.object(
                    tokenizer_utils,
                    "_resolve_local_or_cached_file",
                    wraps=tokenizer_utils._resolve_local_or_cached_file,
                ) as resolve,
            ):
                tokenizer = tokenizer_utils.get_tokenizer(
                    self.model, tokenizer_backend="fastokens", **kwargs
                )
            self.assertEqual(load.call_args.kwargs["revision"], expected)
            resolve.assert_any_call(self.model, "tokenizer.json", expected)
            self.assert_checkpoint_ids(tokenizer)


if __name__ == "__main__":
    unittest.main()
