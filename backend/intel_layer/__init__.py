# Ground Station - Intel Layer
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
Intel Layer – SIGINT/GEOINT fusion suite.

Exposes :class:`IntelFusionSuite` as the single entry-point for the
ground-station demodulator pipeline.  The suite orchestrates:

* :class:`~intel_layer.trilateral.TDOAEngine` – spatial localisation
* :class:`~intel_layer.pattern_analyzer.RFPatternAnalyzer` – anomaly detection
* :class:`~intel_layer.gotham_bridge.GothamBridge` – JSON-LD export
"""

import json
import logging

from .gotham_bridge import GothamBridge
from .pattern_analyzer import RFPatternAnalyzer
from .trilateral import TDOAEngine

__all__ = ["IntelFusionSuite", "TDOAEngine", "RFPatternAnalyzer", "GothamBridge"]

logger = logging.getLogger(__name__)


class IntelFusionSuite:
    """
    Orchestrator for the SIGINT/GEOINT intelligence pipeline.

    Instantiate once per decoder worker process, then call
    :meth:`process_intercept` for every decoded packet.
    """

    def __init__(
        self,
        node_id: str = "TACTICAL_EDGE_ALPHA",
        window_size: int = 50,
        z_threshold: float = 3.0,
        propagation_speed: float = 299_792_458.0,
    ):
        """
        Args:
            node_id: Reporting node identifier forwarded to GothamBridge.
            window_size: Rolling window depth for RFPatternAnalyzer.
            z_threshold: Z-score anomaly threshold for RFPatternAnalyzer.
            propagation_speed: Signal propagation speed (m/s) for TDOAEngine.
        """
        self.tdoa = TDOAEngine(propagation_speed=propagation_speed)
        self.analyzer = RFPatternAnalyzer(window_size=window_size, z_threshold=z_threshold)
        self.bridge = GothamBridge(node_id=node_id)

    def process_intercept(
        self,
        telemetry_event: dict,
        multi_station_tdoa_data: dict | None = None,
    ) -> str:
        """
        Main entry-point: fuse telemetry, anomaly detection, and optional
        TDOA localisation into a JSON-LD intelligence package.

        Args:
            telemetry_event: Dict with signal metadata from the demodulator.
                Expected keys (all optional):
                    timestamp       – Unix epoch float
                    signal          – Dict with ``frequency_mhz`` / ``signal_power_dbfs``
                    from_callsign   – Source callsign string
                    hex_data        – Raw payload as hex string
                    frequency_hz    – Centre frequency in Hz

            multi_station_tdoa_data: Optional dict for TDOA localisation:
                    stations        – List of (X, Y, Z) receiver coordinates
                    time_diffs      – TDOA time deltas relative to station[0]

        Returns:
            Serialised JSON-LD intelligence package string.
        """
        signal_meta = telemetry_event.get("signal") or {}

        # 1. RF anomaly analysis
        anomaly_report = self.analyzer.analyze_signal(
            {
                "timestamp": telemetry_event.get("timestamp"),
                "frequency": signal_meta.get("frequency_mhz"),
                "power_dbfs": signal_meta.get("signal_power_dbfs"),
                "id": telemetry_event.get("from_callsign", "unknown"),
            }
        )

        # 2. Optional TDOA spatial localisation
        coords: list | None = None
        if multi_station_tdoa_data:
            try:
                coords = self.tdoa.locate_emitter(
                    stations=multi_station_tdoa_data["stations"],
                    time_diffs=multi_station_tdoa_data["time_diffs"],
                )
            except Exception:
                logger.exception("TDOA localisation failed")

        # 3. Fuse and export as JSON-LD
        telemetry_data = {
            "from_callsign": telemetry_event.get("from_callsign", "UNKNOWN"),
            "frequency_hz": telemetry_event.get("frequency_hz", 0),
            "hex_data": telemetry_event.get("hex_data", ""),
        }

        intel_package = self.bridge.create_intel_package(
            tdoa_coords=coords,
            anomaly_report=anomaly_report,
            telemetry_data=telemetry_data,
        )

        self.bridge.dispatch(intel_package)

        return intel_package

    def process_intercept_as_dict(
        self,
        telemetry_event: dict,
        multi_station_tdoa_data: dict | None = None,
    ) -> dict:
        """
        Convenience wrapper around :meth:`process_intercept` that returns
        a parsed dict instead of a JSON string.
        """
        return json.loads(self.process_intercept(telemetry_event, multi_station_tdoa_data))
