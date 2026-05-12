"""
wm_client - A lightweight WebSocket client for World Model inference server.

This package provides a simple interface to communicate with a World Model
server over WebSocket, sending image frames and receiving predictions.
"""

from wm_client.client import (
    WMClient,
    WMPredictionOutput,
    call_wm_server,
    call_wm_server_async,
)

__version__ = "0.1.0"
__all__ = [
    "WMClient",
    "WMPredictionOutput",
    "call_wm_server",
    "call_wm_server_async",
]
