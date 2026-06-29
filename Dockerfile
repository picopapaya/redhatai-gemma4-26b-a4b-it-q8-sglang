# RedHatAI Gemma 4 26B-A4B-IT (pre-quantized FP8, compressed-tensors) served by SGLang, on the NVIDIA GB10 (DGX Spark).
# Compare with ../google-gemma4-26b-a4b-it which uses the original Google BF16 weights + fp8 on-the-fly.
#
# Model architecture — Gemma 4 27B is a Mixture-of-Experts model:
#   - 26B total parameters, ~3.8B active per token ("26B-A4B" = 26B total, 4B active)
#   - 128 experts per MoE layer; router activates 2 per token
#   - Hybrid: attention layers are dense; FFN layers are MoE
#   - All 26B params must be in VRAM for routing, but compute cost ≈ a 4B dense model
#
# Quantization — why FP8 over NVFP4 on SM121a (GB10):
#   - NVFP4 has NO native FP4 GEMM kernel on SM12x; it falls back to Marlin, which
#     dequantizes FP4 → BF16 inside the kernel, losing the FP4 FLOPS advantage.
#   - FP8 uses the CUTLASS native matmul path on SM121a for dense/attention layers,
#     avoiding the Marlin dequantize overhead that penalises NVFP4.
#   - NOTE: for MoE expert layers, MXFP8 (block-32 OCP format) still falls back to
#     Marlin W8A16 on SM121 in vLLM (TrtLlmFp8ExpertsBase gates on SM_10x only).
#     Standard FP8 (`--quantization fp8`) takes the CUTLASS path even for MoE layers.
#   - Net result: FP8 is faster than NVFP4 on this chip despite the larger weight size.
#   - Max concurrency capped at 4 via --max-running-requests.
#
# Same base image as the GB10 NVFP4 setup: CUDA 13.x is required for sm_121a,
# and Gemma 4 modeling support requires SGLang >= v0.5.11.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.12.post1-cu130
FROM --platform=linux/arm64 ${SGLANG_IMAGE}

ENV MODEL_ID="RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic" \
    HOST="0.0.0.0" \
    PORT="30000" \
    QUANTIZATION="compressed-tensors" \
    KV_CACHE_DTYPE="fp8_e4m3" \
    CONTEXT_LEN="262144" \
    MEM_FRACTION="0.85" \
    MAX_RUNNING_REQUESTS="4" \
    EXTRA_ARGS="" \
    HF_HOME="/root/.cache/huggingface" \
    # Point Triton at CUDA 13.0's ptxas instead of PyTorch's bundled one.
    # The bundled ptxas predates SM_121 and rejects --gpu-name=sm_121a, causing
    # JIT compilation failures for attention and other kernels at runtime.
    # /usr/local/cuda/bin/ptxas (from CUDA 13.0 in this image) knows SM_121a natively.
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# GB10 (SM_121a) Triton MoE kernel config: the default "3072" batch-size entry
# uses BLOCK_SIZE_K=128 + num_stages=3 = 147456 bytes, exceeding the 101376-byte
# shared-memory limit. This hand-tuned config caps that entry at BLOCK_SIZE_K=64
# so every batch size stays under 74 KB. All other entries match the RTX 6000 Ada
# config (same ~99 KB shared-memory budget).
#
# compressed-tensors (RedHatAI/FP8-Dynamic) looks for the per_channel_quant=True
# variant of the filename; the plain fp8_w8a8 name is used by --quantization fp8.
# The _down suffix is the down-projection MoE config (same limits apply).
ARG CFG_DIR=/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0
COPY triton_moe_config.json ${CFG_DIR}/E=128,N=704,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json
COPY triton_moe_config.json ${CFG_DIR}/E=128,N=704,device_name=NVIDIA_GB10,dtype=fp8_w8a8,per_channel_quant=True.json
COPY triton_moe_config.json ${CFG_DIR}/E=128,N=704,device_name=NVIDIA_GB10,dtype=fp8_w8a8,per_channel_quant=True_down.json

EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
