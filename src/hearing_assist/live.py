from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Event
from typing import Any

import numpy as np

from .config import HearingAssistConfig
from .dsp import BlockDiagnostics, HearingAssistProcessor


def _sounddevice() -> Any:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Live audio requires the optional sounddevice/PortAudio package. "
            "Install with: pip install -e '.[live]'"
        ) from exc
    return sd


def list_audio_devices() -> str:
    sd = _sounddevice()
    return str(sd.query_devices())


def _device(value: str | int | None) -> str | int | None:
    if value is None or isinstance(value, int):
        return value
    stripped = value.strip()
    return int(stripped) if stripped.isdigit() else stripped


@dataclass(frozen=True)
class LiveSummary:
    blocks: int
    stream_status_events: int
    fail_safe_blocks: int
    runtime_mean_ms: float
    runtime_p99_ms: float
    deadline_misses: int
    maximum_limiter_reduction_db: float
    cape_reference_block_ratio: float
    cape_action_switches: int


class LiveHearingAssist:
    def __init__(
        self,
        config: HearingAssistConfig,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
    ) -> None:
        self.config = config
        self.processor = HearingAssistProcessor(config)
        self.input_device = _device(input_device)
        self.output_device = _device(output_device)
        self._diagnostics: deque[BlockDiagnostics] = deque(maxlen=50_000)
        self._status_events = 0

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        del time_info
        if status:
            self._status_events += 1
        if frames != self.config.audio.block_size:
            outdata.fill(0.0)
            self._status_events += 1
            return
        try:
            mono = np.asarray(indata[:, 0], dtype=np.float32)
            output, diagnostics = self.processor.process_block(mono)
            if outdata.shape[1] == 1:
                outdata[:, 0] = output
            else:
                outdata[:] = output[:, None]
            self._diagnostics.append(diagnostics)
        except Exception:
            outdata.fill(0.0)
            self._status_events += 1

    def run(self, duration_sec: float | None = None) -> LiveSummary:
        sd = _sounddevice()
        self.processor.reset()
        self._diagnostics.clear()
        self._status_events = 0
        device = (self.input_device, self.output_device)
        print(
            f"live stream: {self.config.audio.sample_rate} Hz, "
            f"block={self.config.audio.block_size}, "
            f"algorithmic latency={self.processor.algorithmic_latency_ms:.2f} ms"
        )
        print("Press Ctrl+C to stop. Start with the physical output volume at minimum.")
        stop = Event()
        try:
            with sd.Stream(
                samplerate=self.config.audio.sample_rate,
                blocksize=self.config.audio.block_size,
                device=device,
                channels=(self.config.audio.input_channels, self.config.audio.output_channels),
                dtype="float32",
                latency="low",
                callback=self._callback,
            ):
                if duration_sec is None:
                    while not stop.wait(0.5):
                        pass
                else:
                    stop.wait(max(0.0, duration_sec))
        except KeyboardInterrupt:
            pass
        return self.summary()

    def summary(self) -> LiveSummary:
        runtimes = np.asarray([item.runtime_ms for item in self._diagnostics], dtype=float)
        reductions = np.asarray(
            [item.limiter_reduction_db for item in self._diagnostics], dtype=float
        )
        actions = [item.cape_action for item in self._diagnostics]
        deadline = self.processor.algorithmic_latency_ms
        return LiveSummary(
            blocks=int(runtimes.size),
            stream_status_events=self._status_events,
            fail_safe_blocks=int(sum(item.fail_safe_triggered for item in self._diagnostics)),
            runtime_mean_ms=float(np.mean(runtimes)) if runtimes.size else 0.0,
            runtime_p99_ms=float(np.percentile(runtimes, 99)) if runtimes.size else 0.0,
            deadline_misses=int(np.sum(runtimes > deadline)),
            maximum_limiter_reduction_db=float(np.max(reductions)) if reductions.size else 0.0,
            cape_reference_block_ratio=float(np.mean([name == "reference" for name in actions]))
            if actions
            else 1.0,
            cape_action_switches=int(
                sum(current != previous for previous, current in zip(actions, actions[1:]))
            ),
        )
