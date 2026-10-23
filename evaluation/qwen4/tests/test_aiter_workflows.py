import pytest
from install_aiter_workflows import render


def test_aiter_callers_use_one_pinned_harness_without_another_nightly():
    sha = "a" * 40
    files = render(sha)
    caller = files["qwen4-downstream.yml"]
    assert f"uses: apinge/sglang/.github/workflows/_qwen4-eval.yml@{sha}" in caller
    assert f"harness_sha: {sha}" in caller
    assert "branches: [qwen38-flash-next-dev]" in caller
    assert "schedule:" not in caller
    assert "pr_number:" in caller and "candidate_sha:" in caller
    assert f"ref: {sha}" in files["qwen4-ci-comments.yml"]
    assert "repository: apinge/sglang" in files["qwen4-ci-comments.yml"]


def test_mutable_harness_ref_is_rejected():
    with pytest.raises(ValueError):
        render("qwen4_v0.5.21")
