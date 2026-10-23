# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, bounded ACLGraphs for the API-process MOSS Local reference encoder."""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)


class MossNPUReferenceEncoderGraphs:
    """Retain true clip lengths while padding to a small, fixed graph set.

    The upstream tokenizer's SDPA attention is retained; the CUDA windowed
    FlashAttention replacement is never installed. Calls are serialized by the
    local reference batcher. Shared-host graph workers are deliberately excluded.
    """

    def __init__(self, tokenizer: torch.nn.Module, *, n_vq: int) -> None:
        self._tokenizer = tokenizer
        self._n_vq = n_vq
        self._device = next(tokenizer.parameters()).device
        frame = int(tokenizer.downsample_rate)
        rate = int(tokenizer.sampling_rate)
        # At most three graphs: (1, 4s), (1, 8s), (2, 4s), rounded to frames.
        # This cap also applies when the existing fine CUDA bucket switches are set.
        buckets = sorted({math.ceil(seconds * rate / frame) * frame for seconds in (4, 8)})
        self._shapes = [(batch, samples) for batch in (1, 2) for samples in buckets if batch * samples <= buckets[-1]]
        self._buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._runner = NPUExactGraphRunner(
            max_graphs=len(self._shapes),
            component_name="MOSS Local reference encoder",
            disable_config_hint="set VLLM_OMNI_MOSS_REF_GRAPHS=0",
        )
        self._stream = None
        self._failed = False

    @property
    def captured(self) -> list[tuple[int, int]]:
        return sorted(self._buffers)

    @property
    def stats(self) -> dict[str, int]:
        return self._runner.stats

    def _check_healthy(self) -> None:
        if self._failed:
            raise RuntimeError(
                "MOSS reference NPUGraph failed; restart the API process with VLLM_OMNI_MOSS_REF_GRAPHS=0"
            )

    @contextlib.contextmanager
    def _context(self) -> Iterator[None]:
        npu = torch.get_device_module("npu")
        with npu.device(self._device):
            if self._stream is None:
                self._stream = npu.Stream(device=self._device)
            current = npu.current_stream(self._device)
            self._stream.wait_stream(current)
            with npu.stream(self._stream), torch.no_grad():
                try:
                    yield
                finally:
                    current.wait_stream(self._stream)

    def _core(self, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self._tokenizer
        dtype = getattr(tokenizer, "compute_dtype", None)
        # Match eager encoding's outer NPU autocast, including its quantizer policy.
        amp = torch.autocast(device_type="npu", dtype=dtype) if dtype in (torch.float16, torch.bfloat16) else None
        with amp if amp is not None else contextlib.nullcontext():
            hidden, hidden_lengths = tokenizer._flatten_channels_for_codec(audio, lengths)
            with tokenizer._codec_inference_autocast():
                for module in tokenizer.encoder:
                    hidden, hidden_lengths = module(hidden, hidden_lengths)
            _, codes, code_lengths = tokenizer.quantizer(hidden.float(), hidden_lengths, self._n_vq)
        return codes, code_lengths

    def _run(
        self, inputs: tuple[torch.Tensor, torch.Tensor], shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_healthy()
        try:
            return self._runner.run("moss_reference_encode", inputs, shape, self._core)
        except RuntimeError:
            # Capture failures can invalidate allocator/RNG state. Replay errors
            # are also terminal: the caller must not retry eagerly in this process.
            self._failed = True
            raise

    def capture(self) -> None:
        self._check_healthy()
        if self._buffers:
            return
        if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
            raise ValueError("MOSS reference ACLGraph requires ASCEND_LAUNCH_BLOCKING to be unset or 0")
        if not self._runner.is_supported():
            return
        npu = torch.get_device_module("npu")
        with self._context():
            # Select graph-capturable ACLNN kernels in this API process only.
            # The codec worker configures its own process separately.
            npu.config.allow_internal_format = False
            npu.set_compile_mode(jit_compile=False)
            for batch, samples in reversed(self._shapes):
                audio = torch.zeros(batch, int(self._tokenizer.number_channels), samples, device=self._device)
                lengths = torch.full((batch,), samples, dtype=torch.long, device=self._device)
                self._run((audio, lengths), (batch, samples))
                self._buffers[(batch, samples)] = (audio, lengths)
        logger.info("MOSS Local reference ACLGraph capture complete: shapes=%s stats=%s", self.captured, self.stats)

    def encode(self, wavs: list[torch.Tensor]) -> list[torch.Tensor | None]:
        self._check_healthy()
        out: list[torch.Tensor | None] = [None] * len(wavs)
        # Invalid or unsupported clips remain the eager processor's responsibility.
        order = sorted(
            (
                i
                for i, wav in enumerate(wavs)
                if wav.ndim == 2 and wav.shape[0] == self._tokenizer.number_channels and wav.shape[-1] > 0
            ),
            key=lambda i: -wavs[i].shape[-1],
        )
        with self._context():
            pos = 0
            while pos < len(order):
                longest = wavs[order[pos]].shape[-1]
                samples = next((s for b, s in sorted(self._buffers) if b == 1 and s >= longest), None)
                if samples is None:
                    pos += 1
                    continue
                batch = max(b for b, s in self._buffers if s == samples)
                rows = order[pos : pos + batch]
                # Prefer the smallest available batch that fits the selected rows.
                batch = min(b for b, s in self._buffers if s == samples and b >= len(rows))
                audio, lengths = self._buffers[(batch, samples)]
                audio.zero_()
                true_lengths = [wavs[i].shape[-1] for i in rows] + [samples] * (batch - len(rows))
                lengths.copy_(torch.tensor(true_lengths, dtype=torch.long))
                for row, index in enumerate(rows):
                    audio[row, :, : wavs[index].shape[-1]].copy_(wavs[index])
                codes, code_lengths = self._run((audio, lengths), (batch, samples))
                codes = codes[:, : len(rows)].cpu()
                code_lengths = code_lengths[: len(rows)].cpu().tolist()
                for row, index in enumerate(rows):
                    # Own the returned CPU storage even for a one-frame clip.
                    out[index] = codes[:, row, : int(code_lengths[row])].transpose(0, 1).long().contiguous()
                pos += len(rows)
        return out
