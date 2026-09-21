"""OptiLign GUI HTTP & Telemetry Server.

Zero-dependency HTTP server using Python's built-in http.server.
Serves web assets and provides real-time JSON REST endpoints for the GUI.
"""

from __future__ import annotations

import json
import mimetypes
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fsoc_pat.gui.gui_bridge import SimulationBridge

STATIC_DIR = Path(__file__).parent / "web"


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving static dashboard files and /api routes."""

    bridge: SimulationBridge  # Set on class or server instance before serving

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        """Add CORS and disable caching so client always gets latest UI assets."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight."""
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        """Route GET requests."""
        try:
            # Force fresh delivery of assets by removing conditional cache headers
            if "If-Modified-Since" in self.headers:
                del self.headers["If-Modified-Since"]
            if "If-None-Match" in self.headers:
                del self.headers["If-None-Match"]

            if self.path == "/" or self.path == "":
                self.path = "/index.html"

            if self.path.startswith("/api/telemetry"):
                self._send_json(self.bridge.get_telemetry_packet())
                return

            if self.path.startswith("/api/environments"):
                self._send_json({"environments": self.bridge.get_recent_environments()})
                return

            if self.path.startswith("/api/reports"):
                self._send_json(self.bridge.get_reports_data())
                return

            # Serve static assets
            super().do_GET()
        except (BrokenPipeError, ConnectionResetError):
            # Client closed connection prematurely (e.g. page refresh / abort)
            pass

    def do_POST(self) -> None:
        """Route POST control requests."""
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len) if content_len > 0 else b"{}"

            try:
                body = json.loads(post_data.decode("utf-8")) if post_data else {}
            except Exception:
                body = {}

            if self.path == "/api/control/jog":
                pan_rate = float(body.get("pan_rate", 0.0))
                tilt_rate = float(body.get("tilt_rate", 0.0))
                self.bridge.set_manual_rates(pan_rate, tilt_rate)
                self._send_json({"status": "ok", "mode": "manual"})

            elif self.path == "/api/control/mode":
                mode = str(body.get("mode", "auto_track"))
                self.bridge.set_control_mode(mode)
                self._send_json({"status": "ok", "mode": self.bridge.control_mode})

            elif self.path == "/api/control/recenter":
                self.bridge.recenter_camera()
                self._send_json({"status": "ok", "recenter": True})

            elif self.path == "/api/control/test_out_of_fov":
                res = self.bridge.trigger_out_of_fov_test()
                self._send_json(res)

            elif self.path == "/api/control/toggle_random_flight":
                enabled = self.bridge.toggle_random_flight()
                self._send_json({"status": "ok", "random_flight_enabled": enabled})

            elif self.path == "/api/control/zoom":
                zoom = float(body.get("zoom", 1.0))
                self.bridge.set_zoom(zoom)
                self._send_json({"status": "ok", "zoom": self.bridge.zoom})

            elif self.path == "/api/control/pause":
                is_running = self.bridge.toggle_pause()
                self._send_json({"status": "ok", "running": is_running})

            elif self.path == "/api/simulation/start":
                self.bridge.start_simulation()
                self._send_json({"status": "ok", "running": True})

            elif self.path == "/api/simulation/reset":
                self.bridge.reset_simulation()
                self._send_json({"status": "ok", "running": False})

            elif self.path == "/api/control/clear_log":
                self.bridge.clear_logs()
                self._send_json({"status": "ok", "cleared": True})

            elif self.path == "/api/control/scenario":
                scen = str(body.get("scenario", "Urban_Link"))
                self.bridge.set_scenario(scen)
                self._send_json({"status": "ok", "scenario": scen})

            elif self.path == "/api/environments/create":
                res = self.bridge.load_custom_environment(body)
                self._send_json(res)

            elif self.path == "/api/environments/tweak":
                res = self.bridge.update_telemetry_parameters(body)
                self._send_json(res)

            elif self.path == "/api/environments/load":
                env_id = str(body.get("id", ""))
                # Find matching environment in recent environments
                recent = self.bridge.get_recent_environments()
                matched = next((e for e in recent if e.get("id") == env_id or e.get("name") == env_id), None)
                if matched:
                    res = self.bridge.load_custom_environment(matched)
                    self._send_json(res)
                else:
                    # Fallback to scenario setter
                    self.bridge.set_scenario(env_id)
                    self._send_json({"status": "ok", "environment": {"name": env_id}})

            else:
                self.send_error(404, "Endpoint not found")

        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, data: dict[str, Any]) -> None:
        """Send a JSON payload."""
        try:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected before payload was completely written
            pass

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP access logs to keep terminal output clean."""
        pass


class OptiLignHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server that suppresses harmless client disconnect tracebacks."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        import sys
        exc_type, _, _ = sys.exc_info()
        if exc_type in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


def create_server(host: str = "127.0.0.1", port: int = 8080, bridge: SimulationBridge | None = None) -> ThreadingHTTPServer:
    """Create configured HTTP server."""
    if bridge is None:
        bridge = SimulationBridge()

    DashboardRequestHandler.bridge = bridge

    # Attempt to bind; if port is busy, increment
    for p in range(port, port + 20):
        try:
            server = OptiLignHTTPServer((host, p), DashboardRequestHandler)
            server.actual_port = p  # type: ignore[attr-defined]
            return server
        except OSError:
            continue

    raise RuntimeError(f"Could not bind to any port in range {port}-{port+20}")


def run_server(host: str = "127.0.0.1", port: int = 8088, bridge: SimulationBridge | None = None) -> None:
    """Run GUI server blocking until interrupted."""
    if bridge is None:
        bridge = SimulationBridge(target_fps=30.0)
        bridge.start_loop()

    server = create_server(host=host, port=port, bridge=bridge)
    actual_port = getattr(server, "actual_port", port)
    print(f"OptiLign Dashboard running at http://{host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop_loop()
        server.shutdown()
        server.server_close()


def main() -> None:
    """CLI entry point for optilign-gui."""
    import argparse
    parser = argparse.ArgumentParser(description="OptiLign FSOC GUI Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8088, help="Port")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()


