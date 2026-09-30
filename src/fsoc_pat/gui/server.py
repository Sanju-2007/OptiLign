"""OptiLign GUI HTTP & Telemetry Server.

Zero-dependency HTTP server using Python's built-in http.server.
Serves web assets and provides real-time JSON REST endpoints for the GUI.
"""

from __future__ import annotations

import base64
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

            if self.path.startswith("/api/reports/export_isro_csv") or self.path.startswith("/api/reports/export_csv"):
                csv_payload = self.bridge.generate_isro_csv_log().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=pat_performance_log.csv")
                self.send_header("Content-Length", str(len(csv_payload)))
                self.end_headers()
                self.wfile.write(csv_payload)
                return

            if self.path.startswith("/api/reports/export_isro_json") or self.path.startswith("/api/reports/export_json"):
                json_data = self.bridge.generate_isro_json_log()
                json_payload = json.dumps(json_data, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", "attachment; filename=pat_performance_log.json")
                self.send_header("Content-Length", str(len(json_payload)))
                self.end_headers()
                self.wfile.write(json_payload)
                return

            if self.path.startswith("/api/reports"):
                self._send_json(self.bridge.get_reports_data())
                return

            if self.path.startswith("/api/benchmark/status"):
                from dataclasses import asdict
                engine = getattr(self.bridge, "latest_benchmark_engine", None)
                metrics = getattr(self.bridge, "latest_benchmark_metrics", None)
                if engine and metrics:
                    self._send_json({
                        "status": "ok",
                        "metrics": asdict(metrics),
                        "latest_frame_base64": engine.get_latest_jpeg_base64(),
                        "total_records": len(engine.records),
                    })
                else:
                    self._send_json({"status": "idle", "metrics": None})
                return

            if self.path.startswith("/api/benchmark/export_csv"):
                engine = getattr(self.bridge, "latest_benchmark_engine", None)
                if engine and engine.records:
                    csv_lines = [
                        "frame_idx,timestamp_s,detected,centroid_u,centroid_v,ground_truth_u,ground_truth_v,pixel_error,cumulative_rmse,intensity,confidence,processing_time_ms,status\n"
                    ]
                    for r in engine.records:
                        u_str = f"{r.centroid_u:.2f}" if r.centroid_u is not None else ""
                        v_str = f"{r.centroid_v:.2f}" if r.centroid_v is not None else ""
                        err_str = f"{r.pixel_error:.2f}" if r.pixel_error is not None else ""
                        csv_lines.append(
                            f"{r.frame_idx},{r.timestamp_s:.3f},{r.detected},{u_str},{v_str},{r.ground_truth_u:.2f},{r.ground_truth_v:.2f},{err_str},{r.cumulative_rmse:.2f},{r.intensity:.1f},{r.confidence:.3f},{r.processing_time_ms:.2f},{r.status}\n"
                        )
                    payload = "".join(csv_lines).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv")
                    self.send_header("Content-Disposition", "attachment; filename=benchmark_performance_log.csv")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self._send_json({"status": "error", "message": "No benchmark records available"}, status=400)
                return

            if self.path.startswith("/api/benchmark/video"):
                video_file = getattr(self.bridge, "latest_benchmark_video_path", None)
                v_path = None
                if video_file and Path(video_file).is_file():
                    v_path = Path(video_file)
                else:
                    candidates = [
                        Path("results/isro_benchmark_sample.mp4"),
                        Path("data/benchmark/isro_benchmark_figure_of_8_square.mp4"),
                        Path("results/test_bench_compat.mp4"),
                    ]
                    for c in candidates:
                        if c.is_file():
                            v_path = c
                            break
                if v_path and v_path.is_file():
                    with open(v_path, "rb") as vf:
                        vbytes = vf.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(vbytes)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(vbytes)
                else:
                    self.send_error(404, "Benchmark video file not found")
                return

            if self.path.startswith("/api/ml/status"):
                self._send_json(self.bridge.get_ml_training_status())
                return

            if self.path.startswith("/api/standard_benchmark/report"):
                ws_root = Path(__file__).resolve().parents[3]
                candidates = [
                    ws_root / "results/standard_benchmark/compliance_report.html",
                    Path("results/standard_benchmark/compliance_report.html"),
                ]
                report_path = next((c for c in candidates if c.is_file()), None)
                if report_path and report_path.is_file():
                    with open(report_path, "rb") as rf:
                        rbytes = rf.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(rbytes)))
                    self.end_headers()
                    self.wfile.write(rbytes)
                else:
                    self.send_error(404, "Compliance report not generated yet")
                return

            if self.path.startswith("/api/standard_benchmark/summary"):
                ws_root = Path(__file__).resolve().parents[3]
                candidates = [
                    ws_root / "results/standard_benchmark/summary.json",
                    Path("results/standard_benchmark/summary.json"),
                ]
                summary_path = next((c for c in candidates if c.is_file()), None)
                if summary_path and summary_path.is_file():
                    with open(summary_path, "r", encoding="utf-8") as sf:
                        sdata = json.load(sf)
                    self._send_json({"status": "ok", "summary": sdata})
                else:
                    self._send_json({"status": "not_found", "message": "Benchmark summary not found"}, status=404)
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

            elif self.path == "/api/control/set_terminals":
                res = self.bridge.set_terminals(
                    body.get("terminal_a", "building"),
                    body.get("terminal_b", "drone"),
                    terminal_a_trajectory=body.get("terminal_a_trajectory"),
                    terminal_a_speed_mps=body.get("terminal_a_speed_mps"),
                    terminal_b_trajectory=body.get("terminal_b_trajectory"),
                    terminal_b_speed_mps=body.get("terminal_b_speed_mps"),
                )
                status_code = 200 if res.get("status") == "ok" else 400
                self._send_json(res, status=status_code)

            elif self.path == "/api/control/set_kinematics":
                res = self.bridge.set_terminal_kinematics(
                    terminal_a_trajectory=body.get("terminal_a_trajectory"),
                    terminal_a_speed_mps=body.get("terminal_a_speed_mps"),
                    terminal_b_trajectory=body.get("terminal_b_trajectory"),
                    terminal_b_speed_mps=body.get("terminal_b_speed_mps"),
                )
                self._send_json(res)

            elif self.path == "/api/control/test_out_of_fov":
                res = self.bridge.trigger_out_of_fov_test()
                self._send_json(res)

            elif self.path == "/api/control/toggle_random_flight":
                enabled = self.bridge.toggle_random_flight()
                self._send_json({"status": "ok", "random_flight_enabled": enabled})

            elif self.path == "/api/control/toggle_ao":
                enabled = self.bridge.toggle_ao()
                self._send_json({"status": "ok", "ao_enabled": enabled})

            elif self.path == "/api/control/handshake_trigger":
                res = self.bridge.trigger_handshake()
                self._send_json({"status": "ok", "handshake": res})

            elif self.path == "/api/control/toggle_solar_glint":
                enabled = self.bridge.toggle_solar_glint()
                self._send_json({"status": "ok", "solar_glint_enabled": enabled})

            elif self.path == "/api/control/toggle_rf_fallback":
                enabled = self.bridge.toggle_rf_fallback()
                self._send_json({"status": "ok", "rf_fallback_enabled": enabled})

            elif self.path == "/api/control/toggle_decoy_filter":
                enabled = self.bridge.toggle_decoy_filter()
                self._send_json({"status": "ok", "carrier_filter_enabled": enabled})

            elif self.path == "/api/control/zoom":
                zoom = float(body.get("zoom", 1.0))
                self.bridge.set_zoom(zoom)
                self._send_json({"status": "ok", "zoom": self.bridge.zoom})

            elif self.path in ("/api/control/target_motion", "/api/control/trajectory"):
                traj = body.get("trajectory", body.get("motion", "figure_of_8"))
                res = self.bridge.set_terminal_kinematics(terminal_b_trajectory=traj)
                self._send_json(res)

            elif self.path == "/api/control/camera_fov":
                fov = body.get("fov", "4x3")
                res = self.bridge.set_camera_fov(fov)
                self._send_json(res)

            elif self.path == "/api/control/camera_type":
                cam_type = body.get("camera_type", "monochrome_fpa")
                res = self.bridge.set_camera_type(cam_type)
                self._send_json(res)

            elif self.path == "/api/control/target_init_loc":
                loc = body.get("location", body.get("init_loc", "random"))
                res = self.bridge.set_target_initial_location(loc)
                self._send_json(res)

            elif self.path == "/api/control/screen_size":
                size_val = body.get("size", "2000x2000")
                res = self.bridge.set_screen_size(size_val)
                self._send_json(res)

            elif self.path == "/api/control/cam_resolution":
                res_val = body.get("resolution", "640x480")
                res = self.bridge.set_camera_resolution(res_val)
                self._send_json(res)

            elif self.path == "/api/control/camera_speed_limit":
                speed = float(body.get("speed_deg_s", body.get("speed", 5.0)))
                res = self.bridge.set_camera_speed_limit(speed)
                self._send_json(res)

            elif self.path == "/api/control/evaluator_mode":
                enabled = bool(body.get("enabled", True))
                res = self.bridge.set_evaluator_mode(enabled)
                self._send_json(res)

            elif self.path == "/api/control/disturbances":
                res = self.bridge.set_disturbance_controls(
                    contrast_reduction=body.get("contrast_reduction"),
                    brightness_reduction=body.get("brightness_reduction"),
                    camera_jitter_px=body.get("camera_jitter_px"),
                    camera_jitter_freq_hz=body.get("camera_jitter_freq_hz"),
                    platform_motion_px=body.get("platform_motion_px"),
                    platform_motion_type=body.get("platform_motion_type"),
                    preset=body.get("preset"),
                    noise_stddev_px=body.get("noise_stddev_px"),
                    enable_salt_pepper=body.get("enable_salt_pepper"),
                    enable_gaussian_noise=body.get("enable_gaussian_noise"),
                    enable_poisson_noise=body.get("enable_poisson_noise"),
                    salt_pepper_ratio=body.get("salt_pepper_ratio"),
                )
                self._send_json(res)

            elif self.path == "/api/control/target_preset":
                res = self.bridge.set_target_properties(
                    shape=body.get("shape"),
                    size_px=body.get("size_px"),
                )
                self._send_json(res)

            elif self.path == "/api/control/pause":
                is_running = self.bridge.toggle_pause()
                self._send_json({"status": "ok", "running": is_running})

            elif self.path == "/api/simulation/start":
                self.bridge.start_simulation()
                self._send_json({"status": "ok", "running": True})

            elif self.path == "/api/simulation/reset":
                self.bridge.reset_simulation()
                self._send_json({"status": "ok", "running": False})

            elif self.path == "/api/ml/train":
                epochs = int(body.get("epochs", 15))
                lr = float(body.get("lr", 1e-3))
                res = self.bridge.train_ml_model(epochs=epochs, lr=lr)
                self._send_json(res)

            elif self.path == "/api/ml/accept_recovery":
                evt_id = str(body.get("event_id", ""))
                res = self.bridge.accept_recovery(evt_id)
                self._send_json(res)

            elif self.path == "/api/ml/upgrade_policy":
                evt_id = str(body.get("event_id", ""))
                res = self.bridge.upgrade_search_policy(evt_id)
                self._send_json(res)

            elif self.path == "/api/ml/dismiss_prompt":
                res = self.bridge.dismiss_recovery_prompt()
                self._send_json(res)

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

            elif self.path == "/api/benchmark/generate":
                motion = str(body.get("motion", "figure_of_8"))
                shape = str(body.get("shape", "square"))
                target_size = int(body.get("target_size", 10))
                speed_limit = float(body.get("pan_tilt_speed_limit_deg_s", body.get("speed_limit_deg_s", 5.0)))
                preset = body.get("preset", None)
                res = self.bridge.generate_benchmark_sample(
                    motion=motion,
                    shape=shape,
                    target_size=target_size,
                    pan_tilt_speed_limit_deg_s=speed_limit,
                    preset=preset,
                    contrast_reduction=float(body.get("contrast_reduction", 0.0)),
                    brightness_reduction=float(body.get("brightness_reduction", 0.0)),
                    camera_jitter_px=float(body.get("camera_jitter_px", 0.0)),
                    platform_motion_px=float(body.get("platform_motion_px", 0.0)),
                )
                self._send_json(res)

            elif self.path == "/api/benchmark/upload":
                file_name = body.get("filename", "uploaded_benchmark.mp4")
                video_b64 = body.get("video_base64", "")
                if "," in video_b64:
                    video_b64 = video_b64.split(",", 1)[1]
                upload_dir = Path("data/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)
                save_path = upload_dir / Path(file_name).name
                with open(save_path, "wb") as f:
                    f.write(base64.b64decode(video_b64))
                self._send_json({"status": "ok", "saved_path": str(save_path), "filename": Path(file_name).name})

            elif self.path == "/api/benchmark/run":
                video_path = str(body.get("video_path", ""))
                gt_csv = body.get("ground_truth_csv", None)
                video_b64 = body.get("video_base64", None)
                file_name = body.get("filename", "uploaded_benchmark.mp4")

                if video_b64:
                    try:
                        upload_dir = Path("data/uploads")
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        save_path = upload_dir / Path(file_name).name
                        if "," in video_b64:
                            video_b64 = video_b64.split(",", 1)[1]
                        with open(save_path, "wb") as f:
                            f.write(base64.b64decode(video_b64))
                        video_path = str(save_path)
                    except Exception as e:
                        self._send_json({"status": "error", "message": f"Failed to save uploaded video: {e}"}, status=400)
                        return

                if not video_path:
                    # Auto-generate or use default sample
                    sample = self.bridge.generate_benchmark_sample()
                    video_path = sample["video_path"]
                    gt_csv = sample["csv_path"]

                try:
                    res = self.bridge.run_video_benchmark(video_path, gt_csv)
                    self._send_json(res)
                except Exception as e:
                    self._send_json({"status": "error", "message": str(e)}, status=400)

            elif self.path == "/api/standard_benchmark/run":
                try:
                    from fsoc_pat.benchmark.standard_benchmark import StandardBenchmarkEngine, StandardBenchmarkConfig
                    dur = float(body.get("duration", 1.0))
                    trials = int(body.get("trials", 1))
                    cfg = StandardBenchmarkConfig(duration_s=dur, num_trials=trials)
                    eng = StandardBenchmarkEngine(cfg)
                    b_res = eng.run()
                    self._send_json({
                        "status": "ok",
                        "overall_verdict": b_res.overall_verdict,
                        "compliance_score_pct": b_res.compliance_score_pct,
                        "passed_trials": b_res.passed_trials,
                        "total_trials": b_res.total_trials,
                        "report_url": "/api/standard_benchmark/report",
                    })
                except Exception as e:
                    self._send_json({"status": "error", "message": str(e)}, status=500)

            else:
                self.send_error(404, "Endpoint not found")

        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        """Send a JSON payload with configurable HTTP status code."""
        try:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
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


