"""Online spectral analysis and harmonic vibration mode identifier.

Detects and tracks periodic mechanical resonance peaks (reaction wheels,
propeller blade pass frequencies, engine rumble) from noisy gyro/encoder
or vision telemetry using sub-bin interpolated FFT and peak picking.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.predictive.predictive_types import HarmonicMode, PredictiveJitterConfig


class HarmonicSpectralAnalyzer:
    """Identifies and tracks mechanical vibration harmonics in real time."""

    def __init__(self, config: PredictiveJitterConfig | None = None) -> None:
        self.config = config or PredictiveJitterConfig()
        self.buffer_len = self.config.history_buffer_len
        self.fs = self.config.sampling_rate_hz

        # Sliding circular sample buffers
        self._pan_buffer: deque[float] = deque(maxlen=self.buffer_len)
        self._tilt_buffer: deque[float] = deque(maxlen=self.buffer_len)

        # Precompute Hanning window for spectral leakage suppression
        self._window = np.hanning(self.buffer_len)
        self._window_coherent_gain = float(np.sum(self._window) / self.buffer_len)

        # Frequency bins for rfft
        self._freqs = np.fft.rfftfreq(self.buffer_len, d=1.0 / self.fs)

        # Tracked modes for both axes
        self.pan_harmonics: list[HarmonicMode] = []
        self.tilt_harmonics: list[HarmonicMode] = []

    def reset(self) -> None:
        """Clear all sample history and tracked harmonic modes."""
        self._pan_buffer.clear()
        self._tilt_buffer.clear()
        self.pan_harmonics.clear()
        self.tilt_harmonics.clear()

    def update(self, jitter_pan_urad: float, jitter_tilt_urad: float) -> tuple[list[HarmonicMode], list[HarmonicMode]]:
        """Add new jitter measurement samples and update identified harmonic modes.

        Args:
            jitter_pan_urad: Current angular jitter in pan (micro-radians).
            jitter_tilt_urad: Current angular jitter in tilt (micro-radians).

        Returns:
            Tuple of (pan_harmonic_modes, tilt_harmonic_modes).
        """
        self._pan_buffer.append(float(jitter_pan_urad))
        self._tilt_buffer.append(float(jitter_tilt_urad))

        n_curr = len(self._pan_buffer)
        if n_curr < 64:
            return self.pan_harmonics, self.tilt_harmonics

        if n_curr < self.buffer_len:
            arr_pan = np.zeros(self.buffer_len, dtype=np.float64)
            arr_tilt = np.zeros(self.buffer_len, dtype=np.float64)
            arr_pan[:n_curr] = np.array(self._pan_buffer)
            arr_tilt[:n_curr] = np.array(self._tilt_buffer)
            self.pan_harmonics = self._identify_peaks(arr_pan)
            self.tilt_harmonics = self._identify_peaks(arr_tilt)
        else:
            self.pan_harmonics = self._identify_peaks(np.array(self._pan_buffer))
            self.tilt_harmonics = self._identify_peaks(np.array(self._tilt_buffer))

        return self.pan_harmonics, self.tilt_harmonics

    def get_recent_samples(self, axis: str = "pan") -> NDArray[np.float64]:
        """Return the recent sample buffer as a numpy array."""
        buf = self._pan_buffer if axis == "pan" else self._tilt_buffer
        return np.array(buf, dtype=np.float64)

    def _identify_peaks(self, signal: NDArray[np.float64]) -> list[HarmonicMode]:
        """Perform windowed FFT and extract dominant resonance peaks with sub-bin interpolation."""
        # Remove DC bias
        sig_ac = signal - np.mean(signal)

        # Apply Hanning window
        windowed = sig_ac * self._window

        # Fast Fourier Transform (real FFT)
        fft_vals = np.fft.rfft(windowed)
        mags = np.abs(fft_vals) / (self.buffer_len * self._window_coherent_gain * 0.5)

        # Mask frequencies outside operational band [min_freq_hz, max_freq_hz]
        valid_mask = (self._freqs >= self.config.min_freq_hz) & (self._freqs <= self.config.max_freq_hz)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 3:
            return []

        # Calculate noise floor (median magnitude in band)
        band_mags = mags[valid_indices]
        noise_floor = float(np.median(band_mags)) if band_mags.size > 0 else 1e-3
        min_prominence = max(0.5, 2.5 * noise_floor)  # Must be at least 2.5x above noise floor

        # Find local maxima in frequency spectrum
        candidates: list[tuple[float, float, float, float]] = []

        for idx in valid_indices:
            if idx <= 1 or idx >= len(mags) - 2:
                continue

            mag_c = mags[idx]
            mag_l = mags[idx - 1]
            mag_r = mags[idx + 1]

            # Local peak condition
            if mag_c > mag_l and mag_c > mag_r and mag_c >= min_prominence:
                # Sub-bin quadratic peak interpolation:
                # delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
                # where alpha = ln(mag_l), beta = ln(mag_c), gamma = ln(mag_r)
                alpha = math.log(max(1e-12, mag_l))
                beta = math.log(max(1e-12, mag_c))
                gamma = math.log(max(1e-12, mag_r))
                denom = alpha - 2.0 * beta + gamma

                if abs(denom) > 1e-12:
                    delta = 0.5 * (alpha - gamma) / denom
                    delta = max(-0.5, min(0.5, delta))
                else:
                    delta = 0.0

                df = self.fs / self.buffer_len
                peak_freq = (idx + delta) * df
                peak_amp = mag_c * math.exp(-0.25 * (alpha - gamma) * delta)
                peak_phase = float(np.angle(fft_vals[idx]))
                snr_db = 20.0 * math.log10(max(1.0, peak_amp / max(1e-4, noise_floor)))

                candidates.append((peak_freq, peak_amp, peak_phase, snr_db))

        # Sort candidate harmonics by amplitude descending
        candidates.sort(key=lambda c: c[1], reverse=True)
        top_peaks = candidates[: self.config.num_harmonics]

        modes = [
            HarmonicMode(
                frequency_hz=p[0],
                amplitude_urad=p[1],
                phase_rad=p[2],
                snr_db=p[3],
            )
            for p in top_peaks
        ]
        return modes
