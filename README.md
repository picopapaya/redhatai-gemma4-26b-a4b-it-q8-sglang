# redhatai-gemma4-26b-a4b-q8-it-sglang

Docker image that runs **Gemma 4 26B-A4B-IT** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

Uses **pre-quantized FP8 weights** from RedHatAI (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`, compressed-tensors format with per-channel scales) served via [SGLang](https://github.com/sgl-project/sglang).

## What it is

Gemma 4 26B-A4B is a Mixture-of-Experts (MoE) language model: it has 26 billion total parameters but only activates about 4 billion per token, giving it the compute cost of a much smaller model while retaining the capacity of a large one.

This image differs from the sibling [`google-gemma4-26b-a4b-it-bf16-sglang`](https://github.com/picopapaya/google-gemma4-26b-a4b-it-bf16-sglang) project: instead of downloading BF16 weights and quantizing at load time, this image pulls weights that are already FP8-quantized, with per-channel calibration scales baked in by RedHatAI.

The image includes a hand-tuned Triton MoE kernel config that keeps shared memory usage within the GB10's hardware limits. Because the `compressed-tensors` format uses per-channel scales for both the up- and down-projection MoE layers, the config is deployed to three filename variants that SGLang looks up at runtime.

## Pre-quantized FP8 (compressed-tensors) vs on-the-fly FP8

**Pros**

- Smaller download (~26 GB vs ~52 GB for BF16 weights).
- Faster startup — no quantization step at load time, so the server is ready sooner.
- Lower peak VRAM during load — no BF16 and FP8 tensors coexisting in memory.
- Per-channel calibration scales computed from representative data → potentially lower quantization error than per-tensor on-the-fly conversion.

**Cons**

- Third-party weights (RedHatAI), not the official Google checkpoint.
- The `compressed-tensors` format is specific to SGLang/vLLM; less portable than plain FP8.
- Quantization format and scales are fixed — cannot be adjusted at launch time.
- Dependent on RedHatAI keeping the weights up to date with Google's checkpoint.

## Requirements

- NVIDIA GB10 / DGX Spark (SM_121a)
- Docker with NVIDIA Container Toolkit
- A Hugging Face token with access to [`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic) (gated — accept the license on the model page first)
- The `llm-net` Docker network: `docker network create llm-net`

## Usage

```bash
HF_TOKEN=hf_xxx docker compose up --build
```

The server starts on port **30000** and exposes an OpenAI-compatible API once the health check passes (allow a few minutes on first run while weights download).

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Hugging Face access token |
| `CONTEXT_LEN` | `262144` | Maximum context length in tokens |
| `MEM_FRACTION` | `0.85` | Fraction of VRAM reserved for the KV cache |
| `EXTRA_ARGS` | *(empty)* | Extra flags passed directly to `sglang.launch_server` |

## License

MIT — see [LICENSE](LICENSE).
