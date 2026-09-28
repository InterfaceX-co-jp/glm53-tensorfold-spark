# TensorFold serving GLM-5.3-Flash on two DGX Sparks (GB10, sm_121, aarch64).
#
# Base: NVIDIA's PyTorch container — the CUDA toolkit, PyTorch and Triton TensorFold's GLM kernels
# are built and tested with upstream. TensorFold itself comes from the pinned submodule plus our
# patches in patches/, so the image is reproducible and our engine changes ship with it.
FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton \
    CUDA_CACHE_PATH=/cache/nv/ComputeCache \
    CUDA_CACHE_MAXSIZE=4294967296
# Every kernel cache lives in the /cache volume (scripts/serve.sh: glm53-tf-cache), so a restart (a new container)
# reuses them: torch cpp extensions (fast prefill, EXL3, KDA), Triton's compiled kernels, and the driver's JIT cache
# of PTX compiled for sm_121 (CUDA_CACHE_PATH; ~/.nv/ComputeCache by default, lost with every `docker rm`; 4 GiB
# instead of the 256 MiB default so PyTorch's and NCCL's PTX-only kernels stay cached). patches/0140.

COPY vendor/TensorFold /src/TensorFold
COPY patches /src/patches
# PATCHES: space-separated patch number prefixes to apply (e.g. "0001 0002 0003"); empty = every patch, in order
ARG PATCHES=""
RUN cd /src/TensorFold \
 && for p in $(ls /src/patches/*.patch 2>/dev/null | sort); do \
      n=$(basename "$p" | cut -c1-4); \
      if [ -n "$PATCHES" ] && ! echo " $PATCHES " | grep -q " $n "; then echo "skipping $p"; continue; fi; \
      echo "applying $p"; git apply --verbose "$p" || patch -p1 --forward < "$p" || exit 1; done \
 && pip install --no-deps /src/TensorFold \
 && pip install "huggingface-hub>=0.24" "tokenizers>=0.20" "safetensors>=0.4" "jinja2>=3.1" \
 && python -c "import tensorfold, torch, triton; print('tensorfold', tensorfold.__file__, 'torch', torch.__version__, 'triton', triton.__version__)"

COPY docker/entrypoint.sh /usr/local/bin/glm53-tf-entrypoint
ENTRYPOINT ["/usr/local/bin/glm53-tf-entrypoint"]
