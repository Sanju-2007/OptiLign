"""Custom Telemetry Engine and Sink Implementations.

Records user-defined telemetry channels at runtime, calculates running
aerospace metrics (RMS error, 99% jitter, link availability), and streams
to file (CSV, JSONL), memory buffers, or network sockets (UDP).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import socket
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from fsoc_pat.telemetry.channel import TelemetryDataType, TelemetrySchema

logger = logging.getLogger(__name__)


class TelemetrySink(ABC):
    """Abstract sink interface for streaming telemetry data."""

    @abstractmethod
    def open(self, schema: TelemetrySchema) -> None:
        """Initialize sink with schema."""
        pass

    @abstractmethod
    def write_record(self, record: dict[str, Any], step: int) -> None:
        """Write a single telemetry record."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Flush and close sink resources."""
        pass


class MemoryBufferSink(TelemetrySink):
    """In-memory telemetry sink with fast rolling buffer and statistical summary."""

    def __init__(self, max_records: int = 100000) -> None:
        self.max_records = max_records
        self.schema: TelemetrySchema | None = None
        self.records: list[dict[str, Any]] = []

    def open(self, schema: TelemetrySchema) -> None:
        self.schema = schema
        self.records = []

    def write_record(self, record: dict[str, Any], step: int) -> None:
        self.records.append(record.copy())

    def close(self) -> None:
        pass

    def get_channel_series(self, channel_name: str) -> np.ndarray:
        """Extract a channel as a numpy array."""
        values = [r.get(channel_name, float("nan")) for r in self.records]
        return np.array(values, dtype=float)

    def compute_summary_statistics(self) -> dict[str, Any]:
        """Compute statistical summary across recorded channels."""
        if not self.records:
            return {"total_records": 0}

        summary: dict[str, Any] = {
            "total_records": len(self.records),
        }

        # Fine error metrics if present
        if "fine_error_urad" in self.records[0]:
            fine_errs = self.get_channel_series("fine_error_urad")
            valid = fine_errs[~np.isnan(fine_errs)]
            if len(valid) > 0:
                summary["fine_error_rms_urad"] = float(np.sqrt(np.mean(valid**2)))
                summary["fine_error_mean_urad"] = float(np.mean(valid))
                summary["fine_error_max_urad"] = float(np.max(valid))
                summary["fine_error_99pct_urad"] = float(np.percentile(valid, 99))

        # Coarse error metrics if present
        if "coarse_error_urad" in self.records[0]:
            coarse_errs = self.get_channel_series("coarse_error_urad")
            valid = coarse_errs[~np.isnan(coarse_errs)]
            if len(valid) > 0:
                summary["coarse_error_rms_urad"] = float(np.sqrt(np.mean(valid**2)))

        # Link availability
        if "in_fine_lock" in self.records[0]:
            locks = [bool(r.get("in_fine_lock", False)) for r in self.records]
            summary["fine_lock_availability_pct"] = float(sum(locks) / len(locks) * 100.0)

        if "link_snr_db" in self.records[0]:
            snrs = self.get_channel_series("link_snr_db")
            valid_snr = snrs[~np.isnan(snrs)]
            if len(valid_snr) > 0:
                summary["mean_snr_db"] = float(np.mean(valid_snr))
                summary["min_snr_db"] = float(np.min(valid_snr))

        return summary


class CSVTelemetrySink(TelemetrySink):
    """Sinks telemetry data directly to a CSV file."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self.file: io.TextIOWrapper | None = None
        self.writer: Any | None = None
        self.schema: TelemetrySchema | None = None
        self.fieldnames: list[str] = []

    def open(self, schema: TelemetrySchema) -> None:
        self.schema = schema
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(schema.channels.keys())
        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def write_record(self, record: dict[str, Any], step: int) -> None:
        if self.writer and self.schema:
            row = {}
            for name, ch in self.schema.channels.items():
                if step % ch.decimation == 0:
                    val = record.get(name, ch.default_value)
                    row[name] = ch.format_value(val)
                else:
                    row[name] = ""
            self.writer.writerow(row)

    def close(self) -> None:
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None


class JSONLTelemetrySink(TelemetrySink):
    """Sinks telemetry stream as JSON-lines."""

    def __init__(self, filepath: str | Path) -> None:
        self.filepath = Path(filepath)
        self.file: io.TextIOWrapper | None = None
        self.schema: TelemetrySchema | None = None

    def open(self, schema: TelemetrySchema) -> None:
        self.schema = schema
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.filepath, "w")

    def write_record(self, record: dict[str, Any], step: int) -> None:
        if self.file and self.schema:
            out: dict[str, Any] = {}
            for name, ch in self.schema.channels.items():
                if step % ch.decimation == 0:
                    out[name] = record.get(name, ch.default_value)
            self.file.write(json.dumps(out) + "\n")

    def close(self) -> None:
        if self.file:
            self.file.flush()
            self.file.close()
            self.file = None


class UDPSink(TelemetrySink):
    """Streams telemetry packets over UDP broadcast for external ground station listeners."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9876) -> None:
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self.schema: TelemetrySchema | None = None

    def open(self, schema: TelemetrySchema) -> None:
        self.schema = schema
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except Exception as e:
            logger.warning("Could not create UDP socket: %s", e)

    def write_record(self, record: dict[str, Any], step: int) -> None:
        if not self.sock:
            return
        try:
            payload = json.dumps(record).encode("utf-8")
            self.sock.sendto(payload, (self.host, self.port))
        except Exception:
            pass

    def close(self) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None


class TelemetryEngine:
    """Central manager orchestrating custom telemetry schemas and sinks."""

    def __init__(self, schema: TelemetrySchema | None = None) -> None:
        self.schema = schema or TelemetrySchema.standard_fsoc_schema()
        self.sinks: list[TelemetrySink] = []
        self._is_open = False
        self._step = 0
        self.memory_sink = MemoryBufferSink()
        self.add_sink(self.memory_sink)

    def add_sink(self, sink: TelemetrySink) -> None:
        """Add an active telemetry sink."""
        self.sinks.append(sink)
        if self._is_open:
            sink.open(self.schema)

    def open(self) -> None:
        """Initialize all registered sinks."""
        self._step = 0
        for sink in self.sinks:
            sink.open(self.schema)
        self._is_open = True

    def record(self, data: dict[str, Any]) -> None:
        """Record a single telemetry frame."""
        if not self._is_open:
            self.open()

        for sink in self.sinks:
            sink.write_record(data, self._step)
        self._step += 1

    def close(self) -> None:
        """Flush and close all sinks."""
        for sink in self.sinks:
            sink.close()
        self._is_open = False

    @property
    def memory_buffer(self) -> MemoryBufferSink:
        """Access the default in-memory buffer."""
        return self.memory_sink

    def get_summary(self) -> dict[str, Any]:
        """Compute statistical summary of recorded telemetry."""
        return self.memory_sink.compute_summary_statistics()
