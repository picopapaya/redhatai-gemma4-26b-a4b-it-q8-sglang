# redhatai-gemma4-26b-a4b-it-q8-sglang

Docker image that runs **Gemma 4 26B-A4B-IT** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

Uses **pre-quantized FP8 weights** from RedHatAI (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`, compressed-tensors format with per-channel scales), served via [SGLang](https://github.com/sgl-project/sglang).

## What this image is

Gemma 4 26B-A4B is a Mixture-of-Experts model: it has 26 billion total parameters, but for any given word it only actually uses about 4 billion of them. That's what "26B-A4B" means — 26B total, ~4B active. The rest sit in memory ready to be picked, but don't add to the compute cost, so it runs like a much smaller model while still having the knowledge of a much bigger one.

The image also ships a hand-tuned kernel config file so the MoE (mixture-of-experts) layers don't overflow this chip's shared-memory limit which is a workaround for a specific hardware constraint, not something you need to think about day to day.

### Pre-quantized (this image) vs on-the-fly quantization

**What you gain by using pre-quantized weights:**
- Smaller download (~26 GB vs ~52 GB for the original unquantized weights).
- Faster startup — no quantization step at load time, so the server is ready sooner.
- Less memory used briefly during startup — this image never has to hold both the original and quantized weights in memory at once.
- The quantization numbers were calculated in advance from representative data, which can be more accurate than quantizing on the fly.

**What you give up:**
- These are third-party weights (from RedHatAI), not the checkpoint Google publishes directly.
- The specific format used (`compressed-tensors`) only works with SGLang/vLLM — less portable than plain FP8.
- The quantization is fixed at download time — you can't adjust how it's done at launch.
- Depends on RedHatAI keeping their weights in sync whenever Google updates the model.

## Configuration

### Fixed configuration

These define what this image is, not how it's tuned. Changing them means you're describing a different image, not adjusting this one.

| Variable | Value | Why it's fixed |
|---|---|---|
| `MODEL_ID` | `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` | This is which model the image downloads and runs — that's the image's whole identity |
| `QUANTIZATION` | `compressed-tensors` | Has to match the format RedHatAI actually published the weights in |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | Tuned to this checkpoint's quantization |

### Tunable via `.env`

These have a default baked into the image, but you can override them per-deployment by setting them in a `.env` file next to `docker-compose.yml`. Docker Compose reads that file automatically and passes the values into the container when it starts — no image rebuild needed, just edit `.env` and restart. Copy `.env.example` to `.env` to get started.

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | *(required — see Requirements)* | Your Hugging Face access token |
| `CONTEXT_LEN` | `262144` | The longest conversation/prompt (in tokens) the server will accept |
| `MEM_FRACTION` | `0.85` | How much of the GPU's memory this server is allowed to claim |
| `MAX_RUNNING_REQUESTS` | `4` | How many requests SGLang will run concurrently |
| `EXTRA_ARGS` | *(empty)* | Extra flags passed straight through to the underlying `sglang.launch_server` command, for anything not covered above |

## Requirements

- NVIDIA GB10 / DGX Spark (SM_121a)
- Docker with NVIDIA Container Toolkit
- A Hugging Face token with access to [`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic) (gated — accept the license on the model page first)
- The `llm-net` Docker network: `docker network create llm-net`

## Usage

```bash
# Prod — pull image from Docker Hub
HF_TOKEN=hf_xxx docker compose up

# Dev — build image locally
HF_TOKEN=hf_xxx docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

The server starts on port **30000** and exposes an OpenAI-compatible API once the health check passes (allow a few minutes on first run while weights download).

## License

MIT — see [LICENSE](LICENSE).
