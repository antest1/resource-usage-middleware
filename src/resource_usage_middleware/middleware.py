"""
FastAPI middleware to measure per-request elapsed time, peak CPU (RSS) memory usage,
peak CPU utilization (%) for the current process, and peak GPU memory usage
for the current process only (per NVIDIA device).

Features
- Adds response headers:
    - X-Elapsed-Time-ms
    - X-Max-RSS-MB
    - X-Max-CPU-Percent
    - X-GPU-Mem-Per-Device-MB (JSON: {device_index: peak_used_mb}) when NVML available
    - X-Max-RSS (human-readable, e.g. '1.23 GB')
    - X-GPU-Mem-Per-Device (human-readable JSON per device)
- Optionally logs the same metrics via standard logging.
- Lightweight async sampler with configurable polling interval.

Requirements
    pip install fastapi uvicorn psutil pynvml

Notes
- GPU metrics require NVIDIA drivers + NVML. If `pynvml` or NVML aren’t available, GPU metrics are skipped gracefully.
- Peak CPU memory is sampled from process RSS; very short-lived spikes between sampling intervals may be missed. Lower the
  `poll_interval_s` for finer resolution (at the cost of more sampling overhead).
- CPU utilization (%): psutil reports process % relative to a single CPU, so values can exceed 100 on multi-core systems
  (up to 100 * num_logical_cpus).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Dict, Optional

import psutil
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# --- Optional NVML (GPU) support ------------------------------------------------
try:
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore
    _NVML_AVAILABLE = False


def _format_bytes(n: int) -> str:
    """Return a human-readable byte size, e.g., '987 B', '12.3 MB', '1.02 GB'."""
    n = max(0, int(n or 0))
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    val = float(n)
    while val >= 1024.0 and i < len(units) - 1:
        val /= 1024.0
        i += 1
    if i == 0:
        return f"{int(val)} {units[i]}"
    elif val >= 100:
        return f"{val:.0f} {units[i]}"
    elif val >= 10:
        return f"{val:.1f} {units[i]}"
    else:
        return f"{val:.2f} {units[i]}"


# ---- NVML helpers --------------------------------------------------------------

def _get_pid_gpu_bytes_per_device(gpu_handles, pid: int) -> Dict[int, int]:
    """
    Return {device_index: used_bytes_for_pid} for the given PID.
    Tries both compute and graphics process lists and sums usage if both present.
    Falls back to 0 per device if unavailable.
    """
    per_dev: Dict[int, int] = {}
    if not _NVML_AVAILABLE or not gpu_handles:
        return per_dev

    # Resolve best-available API variants once
    get_compute = None
    get_graphics = None
    for name in (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ):
        get_compute = getattr(pynvml, name, None) or get_compute
    for name in (
        "nvmlDeviceGetGraphicsRunningProcesses_v3",
        "nvmlDeviceGetGraphicsRunningProcesses_v2",
        "nvmlDeviceGetGraphicsRunningProcesses",
    ):
        get_graphics = getattr(pynvml, name, None) or get_graphics

    for idx, h in enumerate(gpu_handles):
        used = 0
        try:
            if get_compute:
                for p in get_compute(h):  # type: ignore[misc]
                    try:
                        if int(getattr(p, "pid")) == pid:
                            u = int(getattr(p, "usedGpuMemory", 0))
                            if u >= 0 and u < (1 << 60):
                                used += u
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            if get_graphics:
                for p in get_graphics(h):  # type: ignore[misc]
                    try:
                        if int(getattr(p, "pid")) == pid:
                            u = int(getattr(p, "usedGpuMemory", 0))
                            if u >= 0 and u < (1 << 60):
                                used += u
                    except Exception:
                        pass
        except Exception:
            pass

        per_dev[idx] = used
    return per_dev


class ResourceUsageMiddleware(BaseHTTPMiddleware):
    """Measure elapsed time, peak RSS (CPU mem), peak %CPU for this PID, and per-device peak GPU mem (this PID) per request.

    Args:
        app: FastAPI/Starlette app instance.
        poll_interval_s: Sampling interval in seconds for memory polling.
        enable_gpu: Whether to poll GPU memory via NVML when available.
        add_headers: Whether to attach metrics as response headers.
        log_metrics: Whether to log metrics with the standard logger.
        header_prefix: Prefix for custom response headers.
    """

    def __init__(
        self,
        app: FastAPI,
        *,
        poll_interval_s: float = 0.05,
        enable_gpu: bool = True,
        add_headers: bool = False,
        log_metrics: bool = True,
        header_prefix: str = "X-",
    ) -> None:
        super().__init__(app)
        self.poll_interval_s = max(0.005, float(poll_interval_s))
        self.enable_gpu = bool(enable_gpu)
        self.add_headers = bool(add_headers)
        self.log_metrics = bool(log_metrics)
        self.header_prefix = header_prefix
        self._ensure_nvml_initialized()
        self._logger = logging.getLogger(__name__)

    # NVML is process-wide; init once lazily.
    @staticmethod
    def _ensure_nvml_initialized() -> None:
        if not _NVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
        except Exception:
            pass

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        curr_pid = os.getpid()

        # ---- Peak RSS setup
        peak_rss = process.memory_info().rss  # bytes

        # ---- Peak CPU % setup (this PID)
        # Prime the cpu_percent() internal counters so subsequent non-blocking reads are meaningful
        try:
            process.cpu_percent(None)
        except Exception:
            pass
        peak_cpu_pct: float = 0.0

        # ---- GPU setup (this PID)
        gpu_peak: Dict[int, int] = {}  # device_index -> used_bytes
        gpu_handles = []
        if self.enable_gpu and _NVML_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                for i in range(gpu_count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_handles.append(h)
                    gpu_peak[i] = 0
            except Exception:
                gpu_handles = []

        stop_event = asyncio.Event()

        async def sampler() -> None:
            nonlocal peak_rss, peak_cpu_pct, gpu_peak
            # First immediate sample (no initial sleep)
            try:
                rss_now = process.memory_info().rss
                if rss_now > peak_rss:
                    peak_rss = rss_now

                # non-blocking: % since last call to cpu_percent()
                try:
                    cpu_now = float(process.cpu_percent(0.0))
                    if cpu_now > peak_cpu_pct:
                        peak_cpu_pct = cpu_now
                except Exception:
                    pass

                if gpu_handles:
                    per_dev_now = _get_pid_gpu_bytes_per_device(gpu_handles, curr_pid)
                    for idx, used in per_dev_now.items():
                        if used > gpu_peak.get(idx, 0):
                            gpu_peak[idx] = used
            except Exception:
                pass

            # Periodic samples while the request is in-flight
            interval = self.poll_interval_s
            while not stop_event.is_set():
                try:
                    rss_now = process.memory_info().rss
                    if rss_now > peak_rss:
                        peak_rss = rss_now

                    try:
                        cpu_now = float(process.cpu_percent(0.0))
                        if cpu_now > peak_cpu_pct:
                            peak_cpu_pct = cpu_now
                    except Exception:
                        pass

                    if gpu_handles:
                        per_dev_now = _get_pid_gpu_bytes_per_device(gpu_handles, curr_pid)
                        for idx, used in per_dev_now.items():
                            if used > gpu_peak.get(idx, 0):
                                gpu_peak[idx] = used
                except Exception:
                    pass
                await asyncio.sleep(interval)

        # Launch sampler and handle the request
        task: Optional[asyncio.Task] = asyncio.create_task(sampler())
        try:
            response = await call_next(request)
        finally:
            stop_event.set()
            if task is not None:
                try:
                    await task
                except Exception:
                    pass

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        peak_rss_mb = peak_rss / (1024 ** 2)
        gpu_peak_mb = {k: v / (1024 ** 2) for k, v in gpu_peak.items()}

        peak_rss_human = _format_bytes(peak_rss)
        gpu_peak_human = {k: _format_bytes(int(v_mb * 1024 ** 2)) for k, v_mb in gpu_peak_mb.items()}

        if self.add_headers:
            response.headers[f"{self.header_prefix}Elapsed-Time-ms"] = f"{elapsed_ms:.2f}"
            response.headers[f"{self.header_prefix}Max-RSS-MB"] = f"{peak_rss_mb:.2f}"
            response.headers[f"{self.header_prefix}Max-CPU-Percent"] = f"{peak_cpu_pct:.2f}"
            if gpu_peak_mb:
                response.headers[f"{self.header_prefix}GPU-Mem-Per-Device-MB"] = json.dumps(
                    {str(k): round(v, 2) for k, v in gpu_peak_mb.items()}
                )

            response.headers[f"{self.header_prefix}Max-RSS"] = peak_rss_human
            if gpu_peak_human:
                response.headers[f"{self.header_prefix}GPU-Mem-Per-Device"] = json.dumps(
                    {str(k): v for k, v in gpu_peak_human.items()}, ensure_ascii=False
                )

        if self.log_metrics:
            payload = {
                "path": str(request.url.path),
                "method": request.method,
                "status_code": getattr(response, "status_code", None),
                "elapsed_ms": round(elapsed_ms, 2),
                "peak_rss_mb": round(peak_rss_mb, 2),
                "peak_cpu_percent": round(peak_cpu_pct, 2),
                "gpu_peak_mb": {k: round(v, 2) for k, v in gpu_peak_mb.items()} if gpu_peak_mb else {},
                "peak_rss_human": peak_rss_human,
                "gpu_peak_human": {k: v for k, v in gpu_peak_human.items()} if gpu_peak_human else {},
            }
            self._logger.info("resource_usage: %s", json.dumps(payload, ensure_ascii=False))

        return response
