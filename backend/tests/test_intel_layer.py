# Ground Station - Intel Layer Tests
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
Unit tests for the intel_layer SIGINT/GEOINT fusion suite.
"""

import json
import math
import sys
import os

import pytest

# Ensure the backend package root is on the path when running from the tests/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intel_layer import IntelFusionSuite, GothamBridge, RFPatternAnalyzer, TDOAEngine

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# TDOAEngine
# ---------------------------------------------------------------------------


class TestTDOAEngine:
    def test_locate_emitter_simple_2d(self):
        """Emitter at origin; three receivers in a triangle should converge."""
        engine = TDOAEngine()
        c = engine.c

        # Receivers placed on axes; emitter at (0, 0, 0)
        stations = [(1000.0, 0.0, 0.0), (0.0, 1000.0, 0.0), (-1000.0, 0.0, 0.0)]
        # time_diffs relative to station[0]:
        #   station[0] reference → 0
        #   station[1]: dist_1 - dist_0 / c
        #   station[2]: dist_2 - dist_0 / c
        emitter = [0.0, 0.0, 0.0]
        dists = [math.dist(emitter, s) for s in stations]
        time_diffs = [0.0] + [(dists[i] - dists[0]) / c for i in range(1, len(stations))]

        result = engine.locate_emitter(stations, time_diffs, initial_guess=emitter)
        assert result is not None
        assert len(result) == 3
        for got, exp in zip(result, emitter):
            assert abs(got - exp) < 1.0  # within 1 metre

    def test_locate_emitter_returns_list(self):
        """Return type must be a Python list (JSON-serialisable)."""
        engine = TDOAEngine()
        stations = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0)]
        time_diffs = [0.0, 0.0, 0.0]
        result = engine.locate_emitter(stations, time_diffs)
        assert isinstance(result, list)

    def test_locate_emitter_custom_initial_guess(self):
        """Custom initial_guess is accepted without error."""
        engine = TDOAEngine()
        stations = [(0.0, 0.0, 0.0), (500.0, 0.0, 0.0), (0.0, 500.0, 0.0)]
        time_diffs = [0.0, 0.0, 0.0]
        result = engine.locate_emitter(stations, time_diffs, initial_guess=[250.0, 250.0, 0.0])
        assert result is not None


# ---------------------------------------------------------------------------
# RFPatternAnalyzer
# ---------------------------------------------------------------------------


class TestRFPatternAnalyzer:
    def _build_analyzer_with_baseline(self, n: int = 20):
        """Return an analyzer pre-seeded with *n* stable observations with slight variance."""
        analyzer = RFPatternAnalyzer(window_size=50, z_threshold=3.0)
        for i in range(n):
            # Small deliberate variance so std_dev > 0
            analyzer.analyze_signal(
                {
                    "timestamp": 1_000_000.0 + i * 10,
                    "frequency": 437_500_000.0 + (i % 3) * 1_000.0,
                    "power_dbfs": -50.0 + (i % 3) * 0.1,
                }
            )
        return analyzer

    def test_no_anomaly_stable_signal(self):
        analyzer = self._build_analyzer_with_baseline()
        result = analyzer.analyze_signal(
            {"timestamp": 1_000_200.0, "frequency": 437_500_000.0, "power_dbfs": -50.0}
        )
        assert result["is_anomalous"] is False
        assert result["threat_indicators"] == []

    def test_power_spike_detected(self):
        analyzer = self._build_analyzer_with_baseline()
        result = analyzer.analyze_signal(
            {"timestamp": 1_000_200.0, "frequency": 437_500_000.0, "power_dbfs": 10.0}
        )
        types = [t["type"] for t in result["threat_indicators"]]
        assert "POWER_ANOMALY" in types

    def test_frequency_shift_detected(self):
        analyzer = self._build_analyzer_with_baseline()
        result = analyzer.analyze_signal(
            {"timestamp": 1_000_200.0, "frequency": 999_999_999.0, "power_dbfs": -50.0}
        )
        types = [t["type"] for t in result["threat_indicators"]]
        assert "FREQUENCY_SHIFT" in types

    def test_insufficient_samples_no_anomaly(self):
        """Fewer than 10 samples: no anomaly can be flagged."""
        analyzer = RFPatternAnalyzer()
        for i in range(5):
            result = analyzer.analyze_signal({"frequency": float(i * 1000), "power_dbfs": -40.0})
        assert result["is_anomalous"] is False

    def test_metadata_ref_populated(self):
        analyzer = RFPatternAnalyzer()
        result = analyzer.analyze_signal({"id": "PKT-001", "frequency": 437e6})
        assert result["metadata_ref"] == "PKT-001"

    def test_metadata_ref_defaults_to_unknown(self):
        analyzer = RFPatternAnalyzer()
        result = analyzer.analyze_signal({"frequency": 437e6})
        assert result["metadata_ref"] == "unknown"


# ---------------------------------------------------------------------------
# GothamBridge
# ---------------------------------------------------------------------------


class TestGothamBridge:
    def test_create_intel_package_returns_valid_json(self):
        bridge = GothamBridge(node_id="TEST_NODE")
        anomaly_report = {"is_anomalous": False, "threat_indicators": []}
        telemetry_data = {"from_callsign": "RS0ISS", "frequency_hz": 437_550_000, "hex_data": "deadbeef"}
        pkg_json = bridge.create_intel_package(None, anomaly_report, telemetry_data)
        doc = json.loads(pkg_json)
        assert doc["reportingNode"] == "TEST_NODE"
        assert doc["threatLevel"] == "LOW"
        assert doc["emitterCallsign"] == "RS0ISS"

    def test_create_intel_package_with_coords(self):
        bridge = GothamBridge()
        anomaly_report = {"is_anomalous": False, "threat_indicators": []}
        telemetry_data = {"from_callsign": "SAT1", "frequency_hz": 0, "hex_data": ""}
        pkg_json = bridge.create_intel_package([10.0, 20.0, 100.0], anomaly_report, telemetry_data)
        doc = json.loads(pkg_json)
        assert "location" in doc
        assert doc["location"]["geo:lat"] == 10.0
        assert doc["location"]["geo:long"] == 20.0
        assert doc["location"]["geo:alt"] == 100.0

    def test_threat_level_elevated_for_single_anomaly(self):
        bridge = GothamBridge()
        anomaly_report = {
            "is_anomalous": True,
            "threat_indicators": [{"type": "POWER_ANOMALY", "z_score": 3.5, "value": 5.0}],
        }
        pkg_json = bridge.create_intel_package(None, anomaly_report, {})
        assert json.loads(pkg_json)["threatLevel"] == "ELEVATED"

    def test_threat_level_high_for_multiple_anomalies(self):
        bridge = GothamBridge()
        anomaly_report = {
            "is_anomalous": True,
            "threat_indicators": [
                {"type": "POWER_ANOMALY", "z_score": 4.0, "value": 5.0},
                {"type": "FREQUENCY_SHIFT", "z_score": 4.5, "value": 437e6},
            ],
        }
        pkg_json = bridge.create_intel_package(None, anomaly_report, {})
        assert json.loads(pkg_json)["threatLevel"] == "HIGH"

    def test_jsonld_context_present(self):
        bridge = GothamBridge()
        pkg_json = bridge.create_intel_package(None, {"is_anomalous": False, "threat_indicators": []}, {})
        doc = json.loads(pkg_json)
        assert "@context" in doc
        assert "@id" in doc
        assert "@type" in doc

    def test_dispatch_returns_true(self):
        bridge = GothamBridge()
        assert bridge.dispatch('{"test": true}') is True


# ---------------------------------------------------------------------------
# IntelFusionSuite
# ---------------------------------------------------------------------------


class TestIntelFusionSuite:
    def test_process_intercept_returns_json_string(self):
        suite = IntelFusionSuite()
        result = suite.process_intercept({"timestamp": 1_000_000.0, "signal": {}})
        assert isinstance(result, str)
        doc = json.loads(result)
        assert "@context" in doc

    def test_process_intercept_as_dict_returns_dict(self):
        suite = IntelFusionSuite()
        result = suite.process_intercept_as_dict({"timestamp": 1_000_000.0})
        assert isinstance(result, dict)
        assert "@type" in result

    def test_process_intercept_with_tdoa(self):
        suite = IntelFusionSuite()
        tdoa_data = {
            "stations": [(0.0, 0.0, 0.0), (1000.0, 0.0, 0.0), (0.0, 1000.0, 0.0)],
            "time_diffs": [0.0, 0.0, 0.0],
        }
        result_dict = suite.process_intercept_as_dict(
            {"timestamp": 1_000_000.0, "signal": {}},
            multi_station_tdoa_data=tdoa_data,
        )
        assert "location" in result_dict

    def test_process_intercept_callsign_forwarded(self):
        suite = IntelFusionSuite()
        result_dict = suite.process_intercept_as_dict(
            {"timestamp": 1_000_000.0, "from_callsign": "RS0ISS", "signal": {}}
        )
        assert result_dict["emitterCallsign"] == "RS0ISS"
