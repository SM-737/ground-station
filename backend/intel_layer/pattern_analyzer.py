# Ground Station - Intel Layer: RF Pattern Analyzer
# Developed for the Ground Station project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
RF Pattern Analyzer.

Identifies anomalous RF behavior (frequency shifts, power spikes, timing
anomalies) using a rolling statistical baseline with Z-score detection.
"""

import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

_ANOMALY_TYPES = {
    "frequency": "FREQUENCY_SHIFT",
    "power_dbfs": "POWER_ANOMALY",
    "interval_ms": "TIMING_ANOMALY",
}


class RFPatternAnalyzer:
    """
    Tactical-edge RF Pattern Analyzer.

    Maintains rolling windows of signal metrics and flags observations
    whose Z-score exceeds a configurable threshold.
    """

    def __init__(self, window_size: int = 50, z_threshold: float = 3.0):
        """
        Args:
            window_size: Maximum number of historical samples per metric.
            z_threshold: Z-score magnitude above which a value is anomalous.
        """
        self.window_size = window_size
        self.z_threshold = z_threshold

        self.history: dict[str, deque] = {
            "frequency": deque(maxlen=window_size),
            "power_dbfs": deque(maxlen=window_size),
            "interval_ms": deque(maxlen=window_size),
        }
        self._last_timestamp: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_anomaly(self, metric_name: str, value: float) -> tuple[bool, float]:
        """
        Compute Z-score of *value* against the current rolling window.

        Returns:
            Tuple (is_anomalous, z_score).  Returns (False, 0.0) when the
            window contains fewer than 10 samples or has zero variance.
        """
        data = self.history[metric_name]
        if len(data) < 10:
            return False, 0.0

        arr = np.array(data, dtype=float)
        std_dev = float(np.std(arr))
        if std_dev == 0.0:
            return False, 0.0

        z_score = abs(value - float(np.mean(arr))) / std_dev
        return z_score > self.z_threshold, z_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_signal(self, telemetry_metadata: dict) -> dict:
        """
        Ingest a single packet's metadata and check for anomalies.

        Expected keys in *telemetry_metadata*:
            timestamp   – Unix epoch (float).  Defaults to ``time.time()``.
            frequency   – Centre frequency in Hz (float, optional).
            power_dbfs  – Signal power in dBFS (float, optional).
            id          – Packet identifier for logging (any, optional).

        Returns:
            Dict with keys:
                is_anomalous      – bool
                threat_indicators – list of anomaly dicts
                metadata_ref      – value of telemetry_metadata["id"] or "unknown"
        """
        current_time = float(telemetry_metadata.get("timestamp") or time.time())
        freq = telemetry_metadata.get("frequency")
        power = telemetry_metadata.get("power_dbfs")

        anomalies: list[dict] = []

        # Frequency / Doppler anomaly
        if freq is not None:
            is_anom, z = self._check_anomaly("frequency", float(freq))
            if is_anom:
                anomalies.append(
                    {"type": _ANOMALY_TYPES["frequency"], "z_score": round(z, 2), "value": freq}
                )
            self.history["frequency"].append(float(freq))

        # Power spike / drop anomaly
        if power is not None:
            is_anom, z = self._check_anomaly("power_dbfs", float(power))
            if is_anom:
                anomalies.append(
                    {
                        "type": _ANOMALY_TYPES["power_dbfs"],
                        "z_score": round(z, 2),
                        "value": power,
                    }
                )
            self.history["power_dbfs"].append(float(power))

        # Transmission interval anomaly
        if self._last_timestamp is not None:
            interval = current_time - self._last_timestamp
            is_anom, z = self._check_anomaly("interval_ms", interval)
            if is_anom:
                anomalies.append(
                    {
                        "type": _ANOMALY_TYPES["interval_ms"],
                        "z_score": round(z, 2),
                        "value": interval,
                    }
                )
            self.history["interval_ms"].append(interval)

        self._last_timestamp = current_time

        result = {
            "is_anomalous": len(anomalies) > 0,
            "threat_indicators": anomalies,
            "metadata_ref": telemetry_metadata.get("id", "unknown"),
        }

        if result["is_anomalous"]:
            logger.warning("RF anomaly detected: %s", result["threat_indicators"])

        return result
