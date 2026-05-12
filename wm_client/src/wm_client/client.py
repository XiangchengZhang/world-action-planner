"""
WebSocket client for World Model inference server.

Provides both synchronous and asynchronous interfaces for communicating
with the WM server.
"""

import asyncio
import base64
import io
import json
from collections import namedtuple
from typing import Any, Dict, List, Optional, Union

import imageio.v3 as iio
import numpy as np
import websockets
import websockets.sync.client
from PIL import Image

# Default server configuration
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 7860

# Named tuple for prediction results
WMPredictionOutput = namedtuple(
    "WMPredictionOutput",
    ["full_video", "pred_frames", "pred_panels"],
)


def _encode_image_list(images: List[Any]) -> List[str]:
    """Encode a list of image-like objects into base64-encoded PNG strings.

    Args:
        images: List of numpy arrays or PIL Images.

    Returns:
        List of base64-encoded PNG strings.
    """
    out: List[str] = []
    for arr in images:
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr)
        buf = iio.imwrite("<bytes>", img, extension=".png")
        out.append(base64.b64encode(buf).decode("ascii"))
    return out


def _decode_image_list(b64_list: List[str]) -> List[np.ndarray]:
    """Decode base64-encoded PNG strings into uint8 HxWx3 arrays.

    Args:
        b64_list: List of base64-encoded PNG strings.

    Returns:
        List of numpy arrays (uint8, HxWx3).
    """
    images: List[np.ndarray] = []
    for b64 in b64_list:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        images.append(np.array(img, dtype=np.uint8))
    return images


async def call_wm_server_async(
    history_frames: List[Any],
    history_conds: List[Any],
    future_conds: List[Any],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> Dict[str, Any]:
    """Call the WM WebSocket server asynchronously with in-memory frames.

    Args:
        history_frames: List of numpy arrays or PIL Images for history frames.
        history_conds: List of numpy arrays or PIL Images for history conditions.
        future_conds: List of numpy arrays or PIL Images for future conditions.
        host: Server hostname or IP address.
        port: Server port number.

    Returns:
        Decoded JSON response from the server.
    """
    payload = {
        "history_frames": _encode_image_list(history_frames),
        "history_conds": _encode_image_list(history_conds),
        "future_conds": _encode_image_list(future_conds),
    }

    uri = f"ws://{host}:{port}"
    async with websockets.connect(uri, max_size=None, ping_interval=None) as websocket:
        await websocket.send(json.dumps(payload))
        response_raw = await websocket.recv()
        return json.loads(response_raw)


def call_wm_server(
    history_frames: List[Any],
    history_conds: List[Any],
    future_conds: List[Any],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> Dict[str, Any]:
    """Call the WM WebSocket server synchronously with in-memory frames.

    This is a synchronous wrapper around call_wm_server_async.
    Use this from regular (non-async) code.

    Args:
        history_frames: List of numpy arrays or PIL Images for history frames.
        history_conds: List of numpy arrays or PIL Images for history conditions.
        future_conds: List of numpy arrays or PIL Images for future conditions.
        host: Server hostname or IP address.
        port: Server port number.

    Returns:
        Decoded JSON response from the server.
    """
    return asyncio.run(
        call_wm_server_async(
            history_frames=history_frames,
            history_conds=history_conds,
            future_conds=future_conds,
            host=host,
            port=port,
        )
    )


class WMClient:
    """Convenience client for the WM WebSocket server.

    Maintains a persistent WebSocket connection for multiple predictions.

    Typical usage:
        client = WMClient(host, port)
        result = client.predict(history_frames, history_conds, future_conds)
        result2 = client.predict(...)  # reuses the same connection
        client.close()

    Or use as a context manager:
        with WMClient(host, port) as client:
            result = client.predict(...)

    Args:
        host: Server hostname or IP address.
        port: Server port number.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self._uri = f"ws://{host}:{port}"
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self.connect()

    def connect(self) -> "WMClient":
        """Open the WebSocket connection to the server.

        Returns:
            Self, for method chaining.
        """
        if self._ws is None:
            print(f"Connecting to {self._uri}...")
            self._ws = websockets.sync.client.connect(
                self._uri,
                max_size=None,
                close_timeout=None,
                ping_timeout=None,  # disable ping timeout for long inference
            )
            print(f"Connected to {self._uri}")
        return self

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            self._ws.close()
            self._ws = None
            print("Connection closed")

    def __enter__(self) -> "WMClient":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def predict(
        self,
        history_frames: List[Any],
        history_conds: List[Any],
        future_conds: List[Any],
    ) -> Union[WMPredictionOutput, Dict[str, Any]]:
        """Send one prediction request over the persistent connection.

        Args:
            history_frames: List of numpy arrays or PIL Images for history frames.
            history_conds: List of numpy arrays or PIL Images for history conditions.
            future_conds: List of numpy arrays or PIL Images for future conditions.

        Returns:
            On success (status == "ok"), returns WMPredictionOutput with lists of
            uint8 HxWx3 numpy arrays. On failure, returns the raw error dict
            from the server (expected to contain at least `status` and `message`).
        """
        # Auto-connect if not already connected.
        if self._ws is None:
            self.connect()

        payload = {
            "history_frames": _encode_image_list(history_frames),
            "history_conds": _encode_image_list(history_conds),
            "future_conds": _encode_image_list(future_conds),
        }

        self._ws.send(json.dumps(payload))
        response_raw = self._ws.recv()
        response = json.loads(response_raw)

        if response.get("status") == "ok":
            full_video_b64 = response.get("full_video", [])
            pred_frames_b64 = response.get("pred_frames", [])
            pred_panels_b64 = response.get("pred_panels", [])

            return WMPredictionOutput(
                full_video=_decode_image_list(full_video_b64),
                pred_frames=_decode_image_list(pred_frames_b64),
                pred_panels=[_decode_image_list(pred_panel) for pred_panel in pred_panels_b64],
            )

        # Failure case: return the raw error dict (with status/message)
        return response
