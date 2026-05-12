# WM Client

A lightweight WebSocket client for communicating with a World Model inference server.

## Installation

```bash
pip install wm-client
```

Or install from source:

```bash
cd wm_client
pip install -e .
```

## Dependencies

This package has minimal dependencies:
- `websockets` - WebSocket communication
- `numpy` - Array operations
- `Pillow` - Image processing
- `imageio` - Image encoding/decoding

## Usage

### Basic Usage with Context Manager

```python
from wm_client import WMClient

# Connect to the server
with WMClient(host="localhost", port=7860) as client:
    result = client.predict(
        history_frames=history_frames,  # List of numpy arrays or PIL Images
        history_conds=history_conds,    # List of numpy arrays or PIL Images
        future_conds=future_conds,      # List of numpy arrays or PIL Images
    )
    
    if hasattr(result, 'full_video'):
        print(f"Prediction succeeded: {len(result.full_video)} frames")
    else:
        print(f"Prediction failed: {result}")
```

### Persistent Connection

```python
from wm_client import WMClient

client = WMClient(host="10.31.144.178", port=7860)

# Make multiple predictions with the same connection
result1 = client.predict(history_frames1, history_conds1, future_conds1)
result2 = client.predict(history_frames2, history_conds2, future_conds2)

client.close()
```

### One-shot Synchronous Call

```python
from wm_client import call_wm_server

response = call_wm_server(
    history_frames=history_frames,
    history_conds=history_conds,
    future_conds=future_conds,
    host="localhost",
    port=7860,
)
```

### Async Usage

```python
import asyncio
from wm_client import call_wm_server_async

async def main():
    response = await call_wm_server_async(
        history_frames=history_frames,
        history_conds=history_conds,
        future_conds=future_conds,
        host="localhost",
        port=7860,
    )
    return response

result = asyncio.run(main())
```

## Return Types

### WMPredictionOutput

On successful prediction, `client.predict()` returns a `WMPredictionOutput` namedtuple with:

- `full_video`: List of numpy arrays (uint8, HxWx3) - complete video frames
- `pred_frames`: List of numpy arrays (uint8, HxWx3) - predicted frames only
- `pred_panels`: List of lists of numpy arrays - prediction panels

### Error Response

On failure, returns a dictionary with:
- `status`: Error status string
- `message`: Error message

## Example

```python
import imageio.v3 as iio
from wm_client import WMClient, WMPredictionOutput

# Load video frames
video_frames = iio.imread("video.mp4")
pose_frames = iio.imread("pose_video.mp4")

# Prepare inputs
history_frames = [video_frames[i] for i in range(23)]
history_conds = [pose_frames[i] for i in range(23)]
future_conds = [pose_frames[i] for i in range(23, 63)]

# Make prediction
with WMClient("localhost", 7860) as client:
    result = client.predict(history_frames, history_conds, future_conds)

if isinstance(result, WMPredictionOutput):
    print(f"Got {len(result.full_video)} video frames")
    # Save result video
    import imageio
    imageio.mimwrite("output.mp4", result.full_video, fps=16)
else:
    print(f"Error: {result}")
```

## License

MIT License
