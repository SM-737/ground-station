# Ground Station - Intel Layer: Gotham Bridge
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
Gotham Bridge.

Exports fused SIGINT/GEOINT intelligence as a JSON-LD payload conforming
to a generalized Intelligence/Event ontology for ingestion by external
C2 systems.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_JSONLD_CONTEXT = {
    "@vocab": "https://schema.mil/int#",
    "gotham": "https://palantir.com/ontology/gotham#",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "Event": "gotham:SignalIntercept",
    "location": "geo:SpatialThing",
    "timestamp": "gotham:eventTime",
}


class GothamBridge:
    """
    Exports fused SIGINT/GEOINT data to C2 systems via JSON-LD.

    The generated document conforms to a generalized Intelligence/Event
    ontology suitable for consumption by Palantir GOTHAM or compatible
    tactical C2 ingestion endpoints.
    """

    def __init__(self, node_id: str = "TACTICAL_EDGE_ALPHA"):
        """
        Args:
            node_id: Identifier for the reporting ground-station node.
        """
        self.node_id = node_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_intel_package(
        self,
        tdoa_coords: list | None,
        anomaly_report: dict,
        telemetry_data: dict,
    ) -> str:
        """
        Fuse TDOA location, RF anomaly report, and raw telemetry into a
        JSON-LD document.

        Args:
            tdoa_coords: Optional [X, Y, Z] emitter coordinates from TDOAEngine.
            anomaly_report: Dict returned by RFPatternAnalyzer.analyze_signal.
            telemetry_data: Dict containing signal/telemetry metadata.

        Returns:
            JSON-LD string representation of the intelligence package.
        """
        event_id = f"urn:uuid:{uuid.uuid4()}"
        timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # GEOINT block
        geo_block = None
        if tdoa_coords and len(tdoa_coords) >= 2:
            geo_block = {
                "@type": "location",
                "geo:lat": tdoa_coords[0],
                "geo:long": tdoa_coords[1],
                "geo:alt": tdoa_coords[2] if len(tdoa_coords) > 2 else 0.0,
            }

        # Threat level assessment
        threat_indicators = anomaly_report.get("threat_indicators", [])
        if not anomaly_report.get("is_anomalous"):
            threat_level = "LOW"
        elif len(threat_indicators) == 1:
            threat_level = "ELEVATED"
        else:
            threat_level = "HIGH"

        document: dict = {
            "@context": _JSONLD_CONTEXT,
            "@id": event_id,
            "@type": "Event",
            "timestamp": timestamp_utc,
            "reportingNode": self.node_id,
            "emitterCallsign": telemetry_data.get("from_callsign", "UNKNOWN"),
            "signalFrequency": telemetry_data.get("frequency_hz", 0),
            "threatLevel": threat_level,
            "anomalies": threat_indicators,
            "rawTelemetryPayload": telemetry_data.get("hex_data", ""),
        }

        if geo_block:
            document["location"] = geo_block

        return json.dumps(document, indent=2)

    def dispatch(self, intel_package_json: str) -> bool:
        """
        Dispatch a JSON-LD intelligence package to a C2 endpoint.

        In a production deployment this method should push the payload to
        a Kafka topic, NATS JetStream stream, or REST ingestion endpoint.
        The default implementation logs the package for local inspection.

        Args:
            intel_package_json: Serialised JSON-LD string from
                                 ``create_intel_package``.

        Returns:
            True on success.
        """
        logger.info("Intel package ready for dispatch:\n%s", intel_package_json)
        return True
