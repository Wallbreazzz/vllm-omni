# MOSS Local reference encoding on Ascend

MOSS-TTS Local can optionally capture its API-process reference audio encoder
with ACLGraph. NPU eager placement must already be available. This accelerates
the reference tokenizer, separately from the talker and streaming decoder graphs.
Hardware performance and audio quality must be validated on the deployment's
model, tokenizer and Ascend software stack.

## Enable reference graphs

Set the existing reference graph switch before starting the server:

```bash
export VLLM_OMNI_MOSS_REF_GRAPHS=1
```

The default is `0` (eager). No request fields change. The tokenizer uses its
upstream SDPA attention; CUDA FlashAttention patches and `torch.compile` are not
used on this path. Capture selects `allow_internal_format=False` and
`jit_compile=False` in the API process. `ASCEND_LAUNCH_BLOCKING=1` is incompatible
with capture and is rejected.

The initial implementation captures at most three frame-aligned shapes:
one clip at four seconds, one clip at eight seconds, and two clips at four
seconds. Larger batches are split. Clip lengths remain inputs to the graph,
and outputs are cropped to the tokenizer's actual code lengths. Clips beyond
eight seconds use eager encoding. CUDA fine-bucket, compilation and batched
transfer options do not expand this NPU graph set.

Shared-host reference encoding remains eager on NPU. Graphs currently serve
only the ordinary local encoder, where the existing batcher serializes calls.
Capture and replay establish an explicit NPU device and stream context on their
calling thread. A graph failure is terminal for that encoder: restart the API
process with `VLLM_OMNI_MOSS_REF_GRAPHS=0` to use eager execution. Continuing with
eager kernels in a process after a failed capture can be unsafe.

## Unit tests

With the project's test dependencies installed, the new unit tests run on CPU
and mock the NPU runtime; they do not establish device-kernel correctness:

```bash
python -m pytest \
  tests/platforms/npu/test_moss_tts_reference_encoder_graphs.py \
  tests/model_executor/models/moss_tts/test_reference_encoder.py \
  --run-level core_model -m 'core_model and cpu' \
  -k 'reference_aclgraph or npu_reference_graph' -q
```

## Hardware comparison

Stop other workloads on the assigned NPU. Use an Ascend environment with
`torch_npu`, model/tokenizer dependencies, and complete locally available
checkpoints. Both `config.json` and `processor_config.json` of the Local model
must reference the local tokenizer when Hugging Face access is unavailable.
The files are not rewritten by the benchmark.

```bash
python benchmarks/tts/benchmark_moss_reference_graphs.py \
  --model /path/to/local-model \
  --references /path/to/reference1.wav /path/to/reference2.wav \
  --device npu:0 --repeats 5 --output /path/to/new-results
```

The benchmark runs eager and graph modes in separate processes using identical
graph-compatible kernel settings. It bypasses result caches and reports:

- Capture/preparation time, warm uncached encoding median and range, and NPU memory.
- Captured shapes and actual replay hits, to detect accidental eager fallback.
- Eager/graph code comparisons, repeated-input stability, mixed batches and a
  reference longer than all buckets.
- A same-padded-input eager/graph comparison to separate shape-padding effects
  from capture/replay effects.

Model loading and capture are excluded from warm timings and reported separately
where applicable. File decoding is outside encoding timings. Preprocessing and
transfers are included. This is not an end-to-end speech-serving benchmark.
Add `--profile` to export an additional untimed CPU/NPU Chrome trace per backend
for inspecting host launches and device operator costs.
Differences in valid code lengths require investigation. Different code values
also require paired speech-generation/listening checks before enabling graphs
in a service; the benchmark does not certify voice-cloning quality.
