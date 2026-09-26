"""
Optional diagnostic helper for BizIntel logging experiments.

This file is not imported by the production app. Keep backend/main.py as the
single production FastAPI entry point.

Do not store API keys or secrets here. Use backend/.env locally and Render
Environment Variables in production.
"""

import logging
from typing import Any

logger = logging.getLogger("bizintel.diagnostics")


def log_upload_checkpoint(message: str, **metadata: Any) -> None:
    """Log a local upload debugging checkpoint."""
    logger.info(
        message,
        extra={
            "event": "upload_checkpoint",
            **metadata,
        },
    )
