"""
HWGAT Client Bridge for InclusiveMatch AI WebRTC Calls
======================================================
Connects the Flask application and WebRTC video call system to the isolated
HWGAT (Hierarchical Windowed Graph Attention Network) sign-recognition service.

Provides:
- Thread-safe HTTP communication with the HWGAT microservice on 127.0.0.1:5055.
- Cached health monitoring with auto-reconnect.
- Seamless fallback to production GCN-T model if HWGAT service is stopped or restarting.
- Non-blocking timeouts to protect WebRTC call latency.
"""

import logging
import os
import time
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger("ai_modules.hwgat_client")

# Default service URL on local loopback
DEFAULT_HWGAT_URL = os.environ.get("HWGAT_SERVICE_URL", "http://127.0.0.1:5055")


class HWGATClient:
    """Singleton client managing communication with the HWGAT microservice."""
    _instance: Optional["HWGATClient"] = None

    def __init__(self, service_url: str = DEFAULT_HWGAT_URL):
        self.service_url = service_url.rstrip("/")
        self._last_health_check = 0.0
        self._is_healthy = False
        self._info: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "HWGATClient":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_available(self, force_refresh: bool = False) -> bool:
        """Check if HWGAT service is running and healthy (cached for 10s)."""
        now = time.time()
        if not force_refresh and (now - self._last_health_check) < 10.0:
            return self._is_healthy

        self._last_health_check = now
        try:
            resp = requests.get(f"{self.service_url}/health", timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ready":
                    self._is_healthy = True
                    self._info = data
                    return True
        except Exception:
            pass

        self._is_healthy = False
        self._info = {}
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """Returns details about the active HWGAT model."""
        if not self._info:
            self.is_available(force_refresh=True)
        return self._info

    def recognize_frame(
        self,
        frame_bytes: bytes,
        user_id: str = "anonymous",
        threshold: float = 0.25,
        timeout: float = 3.5,
    ) -> Optional[Dict[str, Any]]:
        """
        Forward frame binary to the HWGAT service for real-time inference.
        Returns None if service is unavailable, triggering fallback.
        """
        if not self.is_available():
            return None

        try:
            files = {
                "frame": ("frame.jpg", frame_bytes, "image/jpeg"),
            }
            data = {
                "user_id": user_id,
                "threshold": str(threshold),
            }
            resp = requests.post(
                f"{self.service_url}/recognize_frame",
                files=files,
                data=data,
                timeout=timeout,
            )
            if resp.status_code == 200:
                result = resp.json()
                result["engine"] = "HWGAT"
                result["vocabulary_size"] = self._info.get("classes_count", 2002)
                return result
        except requests.exceptions.Timeout:
            logger.warning("[HWGATClient] Frame recognition timed out (> %ss)", timeout)
        except Exception as exc:
            logger.warning("[HWGATClient] Error communicating with HWGAT: %s", exc)
            self._is_healthy = False

        return None

    def reset_user(self, user_id: str) -> bool:
        """Clear user landmark queue and detector state on service."""
        try:
            resp = requests.post(
                f"{self.service_url}/reset_user",
                data={"user_id": user_id},
                timeout=1.5,
            )
            return resp.status_code == 200
        except Exception:
            return False

