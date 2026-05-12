from __future__ import annotations

import asyncio
import argparse
import base64
import io
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
import websockets
from websockets.server import WebSocketServerProtocol

from pipeline import VideoPredictionPipeline, PipelineVideoOutput

DEFAULT_CHECKPOINT_PATH = str(
    Path(__file__).resolve().parent / "data/ckpts/libero_90_base/checkpoints/latest.ckpt"
)


# =========================
# 1) Utilities
# =========================
def _ensure_uint8_rgb(x: Any) -> np.ndarray:
    """Accept PIL or numpy, return uint8 HxWx3."""
    if isinstance(x, Image.Image):
        arr = np.array(x.convert("RGB"), dtype=np.uint8)
    elif isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
    else:
        raise TypeError(f"Unsupported image type: {type(x)}")

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3, got {arr.shape}")
    return arr


def _decode_image_list(b64_list: List[str]) -> List[np.ndarray]:
    """Decode a list of base64-encoded PNG images into numpy arrays."""
    images: List[np.ndarray] = []
    for b64 in b64_list:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        images.append(_ensure_uint8_rgb(img))
    return images


def _encode_image_list(images: List[Any]) -> List[str]:
    """Encode a list of image-like objects into base64-encoded PNG strings."""
    out: List[str] = []
    for img in images or []:
        arr = _ensure_uint8_rgb(img)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


# =========================
# 2) WebSocket server wrapper
# =========================
@dataclass
class ServerConfig:
    # Host to bind the WebSocket server on (e.g., 0.0.0.0 to listen on all interfaces).
    host: str = "0.0.0.0"
    port: int = 7860


class WMWebSocketServer:
    """WebSocket server that exposes the VideoPredictionPipeline.

    Protocol (JSON over WebSocket):
    Request:
      {
        "history_frames": ["<b64 png>", ...],
        "history_conds":  ["<b64 png>", ...],
        "future_conds":   ["<b64 png>", ...]
      }

    Response on success:
      {
        "status": "ok",
        "full_video":  ["<b64 png>", ...],
        "pred_frames": ["<b64 png>", ...],
        "pred_panels": ["<b64 png>", ...]
      }

    Response on error:
      {
        "status": "error",
        "message": "..."
      }
    """

    def __init__(self, pipeline: Any, cfg: ServerConfig):
        self.pipeline = pipeline
        self.cfg = cfg

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        import json

        # Log when a new WebSocket connection is established.
        remote = websocket.remote_address
        print(f"New WebSocket connection from {remote} -> ws://{self.cfg.host}:{self.cfg.port}")

        async for message in websocket:
            try:
                payload = json.loads(message)

                hf_b64 = payload.get("history_frames", [])
                hc_b64 = payload.get("history_conds", [])
                fc_b64 = payload.get("future_conds", [])

                hf = _decode_image_list(hf_b64)
                hc = _decode_image_list(hc_b64)
                fc = _decode_image_list(fc_b64)

                pred = self.pipeline(hf, hc, fc)

                if not isinstance(pred, PipelineVideoOutput):
                    raise ValueError("Pipeline must return PipelineVideoOutput class.")

                pred_dict = pred._asdict()

                response = {
                    "status": "ok",
                    "full_video": _encode_image_list(pred_dict.get("full_video", [])),
                    "pred_frames": _encode_image_list(pred_dict.get("pred_frames", [])),
                    "pred_panels": [_encode_image_list(panels) for panels in pred_dict.get("pred_panels", [])],
                }
            except Exception as e:
                traceback.print_exc()
                response = {
                    "status": "error",
                    "message": str(e) or repr(e),
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }

            await websocket.send(json.dumps(response))

    async def serve_forever(self) -> None:
        async with websockets.serve(
            self._handle_connection,
            self.cfg.host,
            self.cfg.port,
            max_size=None,   # allow arbitrarily large messages
            ping_interval=None,  # disable built-in keepalive timeouts
            ping_timeout=None,   # disable ping timeout for long inference
        ):
            # Determine a user-friendly IP/hostname to display, even if bound to 0.0.0.0.
            import socket

            display_host = os.environ.get("PUBLIC_HOST") or os.environ.get("HOST")
            if not display_host or display_host == "0.0.0.0":
                try:
                    display_host = socket.gethostbyname(socket.gethostname())
                except Exception:
                    display_host = self.cfg.host

            print(f"WebSocket server listening on ws://{display_host}:{self.cfg.port}")
            await asyncio.Future()  # run forever


# =========================
# 3) Entrypoint
# =========================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the world model WebSocket server.")
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("WORLD_MODEL_PATH", DEFAULT_CHECKPOINT_PATH),
        help="Checkpoint path or run id used to load the world model pipeline.",
    )
    parser.add_argument(
        "--hist-guidance",
        type=float,
        default=0.0,
        help="Override for algorithm.hist_guidance.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=20,
        help="Override for algorithm.sample_steps.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind the WebSocket server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "7880")),
        help="Port to bind the WebSocket server to.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    overrides = [
        f"algorithm.hist_guidance={args.hist_guidance}",
        f"algorithm.sample_steps={args.sample_steps}",
        "algorithm.model.compile=false",
        "algorithm.vae.compile=false",
        "algorithm.text_encoder.compile=false",
    ]
    pipeline = VideoPredictionPipeline.from_pretrained(args.checkpoint_path, overrides=overrides)
    cfg = ServerConfig(
        host=args.host,
        port=args.port,
    )

    server = WMWebSocketServer(pipeline, cfg)
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
