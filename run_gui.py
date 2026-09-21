#!/usr/bin/env python3
"""OptiLign — Launch Interactive FSOC Virtual Camera Tracking Simulator.

Starts the simulation bridge and serves the interactive HUD web dashboard.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser

from fsoc_pat.gui.gui_bridge import SimulationBridge
from fsoc_pat.gui.server import create_server


def main() -> None:
    parser = argparse.ArgumentParser(description="OptiLign FSOC Virtual Camera Tracking Simulator GUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8088, help="Port to bind (default: 8088)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    print("=" * 70)
    print("  OptiLign — FSOC Virtual Camera Tracking Simulator")
    print("=" * 70)

    # 1. Initialize simulation bridge
    print("▶ Initializing simulation subsystems & closed-loop engine...")
    bridge = SimulationBridge(target_fps=30.0)
    bridge.start_loop()
    print("  ✓ Virtual Gimbal (2-DOF pan/tilt with actuator lag)")
    print("  ✓ Optical Perception Pipeline (synthetic frames + detector)")
    print("  ✓ Kalman Filter (Constant Velocity motion model)")
    print("  ✓ Closed-Loop PID Controller with anti-windup")

    # 2. Start HTTP & Telemetry Server
    server = create_server(host=args.host, port=args.port, bridge=bridge)
    actual_port = getattr(server, "actual_port", args.port)
    url = f"http://{args.host}:{actual_port}"

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"\n▶ Dashboard Server running at:")
    print(f"  ➜  {url}")
    print("\n  Press Ctrl+C to stop the simulator.")
    print("=" * 70)

    # 3. Open browser
    if not args.no_browser:
        def _open():
            time.sleep(0.6)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n▶ Shutting down simulator...")
        bridge.stop_loop()
        server.shutdown()
        server.server_close()
        print("  ✓ Simulator stopped cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
