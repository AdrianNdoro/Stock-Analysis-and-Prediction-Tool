"""Container healthcheck: exit 0 if Streamlit answers on its health endpoint, else 1."""

import os
import sys
import urllib.request

port = os.environ.get("PORT", "8501")
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=3) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:  # noqa: BLE001 - any failure means unhealthy
    sys.exit(1)
