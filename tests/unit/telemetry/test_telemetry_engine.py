"""Unit tests for the Telemetry Engine and Custom Channels."""

import json
from pathlib import Path
import pytest
import numpy as np

from fsoc_pat.telemetry.channel import TelemetryChannel, TelemetryDataType, TelemetrySchema
from fsoc_pat.telemetry.engine import (
    CSVTelemetrySink,
    JSONLTelemetrySink,
    MemoryBufferSink,
    TelemetryEngine,
    UDPSink,
)


def test_telemetry_channel_formatting():
    ch_float = TelemetryChannel(name="fine_err", unit="urad", precision=3)
    assert ch_float.format_value(1.234567) == "1.235"
    assert ch_float.format_value(None) == ""

    ch_int = TelemetryChannel(name="step", data_type=TelemetryDataType.INT)
    assert ch_int.format_value(42.9) == "42"

    ch_bool = TelemetryChannel(name="lock", data_type=TelemetryDataType.BOOL)
    assert ch_bool.format_value(True) == "1"
    assert ch_bool.format_value(False) == "0"

    ch_vec = TelemetryChannel(name="pos", data_type=TelemetryDataType.VECTOR3, precision=1)
    assert ch_vec.format_value([1.0, 2.0, 3.0]) == "[1.0, 2.0, 3.0]"


def test_standard_fsoc_schema():
    schema = TelemetrySchema.standard_fsoc_schema(sampling_frequency_hz=100.0)
    assert "fine_error_urad" in schema.channels
    assert "coarse_error_urad" in schema.channels
    assert "fsm_pan_urad" in schema.channels
    assert schema.sampling_frequency_hz == 100.0


def test_memory_buffer_sink_and_statistics():
    schema = TelemetrySchema.standard_fsoc_schema()
    engine = TelemetryEngine(schema)
    engine.open()

    for i in range(100):
        engine.record({
            "time_s": i * 0.01,
            "step": i,
            "fine_error_urad": 0.5 + 0.1 * np.sin(i),
            "coarse_error_urad": 15.0 + np.cos(i),
            "in_fine_lock": True if i > 10 else False,
            "link_snr_db": 24.5,
        })
    engine.close()

    summary = engine.get_summary()
    assert summary["total_records"] == 100
    assert 0.4 < summary["fine_error_rms_urad"] < 0.7
    assert summary["fine_lock_availability_pct"] == 89.0
    assert summary["mean_snr_db"] == 24.5


def test_csv_and_jsonl_sinks(tmp_path: Path):
    csv_path = tmp_path / "telemetry.csv"
    jsonl_path = tmp_path / "telemetry.jsonl"

    schema = TelemetrySchema()
    schema.add_channel("time_s", unit="s", precision=2)
    schema.add_channel("val_a", unit="m", precision=1)
    schema.add_channel("active", data_type=TelemetryDataType.BOOL)

    engine = TelemetryEngine(schema)
    engine.add_sink(CSVTelemetrySink(csv_path))
    engine.add_sink(JSONLTelemetrySink(jsonl_path))

    engine.open()
    engine.record({"time_s": 0.1, "val_a": 12.34, "active": True})
    engine.record({"time_s": 0.2, "val_a": 15.67, "active": False})
    engine.close()

    assert csv_path.exists()
    assert jsonl_path.exists()

    with open(csv_path) as f:
        lines = [l.strip() for l in f.readlines()]
    assert lines[0] == "time_s,val_a,active"
    assert lines[1] == "0.10,12.3,1"
    assert lines[2] == "0.20,15.7,0"

    with open(jsonl_path) as f:
        jlines = [json.loads(l) for l in f.readlines()]
    assert len(jlines) == 2
    assert jlines[0]["time_s"] == 0.1
    assert jlines[0]["active"] is True
