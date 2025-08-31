# resource-usage-middleware

FastAPI middleware to measure per-request elapsed time, peak CPU (RSS) memory usage,
and peak GPU memory usage (per NVIDIA device) during the lifecycle of the request.

## Features
- Adds response headers:
    - X-Elapsed-Time-ms
    - X-Max-RSS-MB
    - X-GPU-Mem-Per-Device-MB (JSON: {device_index: peak_used_mb}) when NVML available
- Optionally logs the same metrics via standard logging.
- Lightweight async sampler with configurable polling interval.

## Requirements
    pip install fastapi uvicorn psutil pynvml

## Example Usage

You can drop this file anywhere in your project and register the middleware like so:

```python
from fastapi import FastAPI
from resource_usage_middleware import ResourceUsageMiddleware

app = FastAPI()
app.add_middleware(
    ResourceUsageMiddleware,
    poll_interval_s=0.02,   # finer sampling (optional)
    enable_gpu=True,        # disable if you don't care about GPU mem
    add_headers=True,       # attach metrics to response headers
    log_metrics=True,       # also log metrics
    header_prefix="X-",     # customize header prefix if needed
)

@app.get("/ping")
async def ping():
    return {"ok": True}
```

## Notes
- GPU metrics require NVIDIA drivers + NVML. If `pynvml` or NVML aren’t available, GPU metrics are skipped gracefully.
- Peak CPU memory is sampled from process RSS; very short-lived spikes between sampling intervals may be missed. Lower the
  `poll_interval_s` for finer resolution (at the cost of more sampling overhead).