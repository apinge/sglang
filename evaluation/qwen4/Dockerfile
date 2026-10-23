ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV GPU_ARCHS=gfx942 PYTORCH_ROCM_ARCH=gfx942 SGLANG_BUILD_RUST_EXTS=none
ENV QWEN4_EVALUATION_ROOT=/ci/evaluation
COPY build_install.py /ci/build_install.py
COPY plan.json /ci/build-plan.json
COPY retry-constraints.txt /ci/retry-constraints.txt
RUN python /ci/build_install.py /ci/build-plan.json
COPY harness /ci/harness
COPY scripts /ci/evaluation
WORKDIR /ci
ENTRYPOINT []
CMD ["sleep", "infinity"]
