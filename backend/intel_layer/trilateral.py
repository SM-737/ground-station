# Ground Station - Intel Layer: TDOA Trilateration Engine
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
TDOA Trilateration Engine.

Resolves emitter spatial coordinates from Time Difference of Arrival (TDOA)
measurements recorded at spatially separated receiver stations.
"""

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class TDOAEngine:
    """
    Trilateration / TDOA Solver Engine.

    Converts nanosecond-precise timestamp deltas from spatially separated
    receivers into estimated Cartesian (X, Y, Z) coordinates using
    non-linear optimization.
    """

    def __init__(self, propagation_speed: float = 299_792_458.0):
        """
        Args:
            propagation_speed: Signal propagation speed in m/s.
                               Defaults to the speed of light in vacuum.
        """
        self.c = propagation_speed

    def _tdoa_error(
        self,
        guess_coords: np.ndarray,
        station_coords: np.ndarray,
        time_diffs: np.ndarray,
        ref_idx: int = 0,
    ) -> float:
        """
        Objective function for the L-BFGS-B optimizer.

        Computes the sum of squared differences between expected TDOA
        (derived from the guessed emitter position) and the measured TDOA.

        Args:
            guess_coords: Candidate emitter position [X, Y, Z].
            station_coords: Array of receiver positions, shape (N, 3).
            time_diffs: Time deltas (seconds) relative to reference station.
                        time_diffs[ref_idx] must be 0.0.
            ref_idx: Index of the reference station.

        Returns:
            Scalar residual error.
        """
        ref_station = station_coords[ref_idx]
        expected_dist_ref = np.linalg.norm(guess_coords - ref_station)

        error = 0.0
        for i, station in enumerate(station_coords):
            if i == ref_idx:
                continue
            expected_dist_i = np.linalg.norm(guess_coords - station)
            expected_diff = expected_dist_i - expected_dist_ref
            measured_diff = time_diffs[i] * self.c
            error += (expected_diff - measured_diff) ** 2

        return error

    def locate_emitter(
        self,
        stations: list,
        time_diffs: list,
        initial_guess: list | None = None,
    ) -> list | None:
        """
        Resolve the emitter location from multi-station TDOA measurements.

        Args:
            stations: List of (X, Y, Z) receiver coordinates.  Minimum 3.
            time_diffs: Time deltas (seconds) relative to station[0].
                        time_diffs[0] must be 0.0.
            initial_guess: Optional starting (X, Y, Z) for the solver.
                           Defaults to the centroid of the receiver array.

        Returns:
            Estimated [X, Y, Z] of the emitter, or None if optimization
            failed to converge.
        """
        station_coords = np.array(stations, dtype=float)
        td = np.array(time_diffs, dtype=float)

        if initial_guess is None:
            initial_guess = np.mean(station_coords, axis=0)

        result = minimize(
            self._tdoa_error,
            np.array(initial_guess, dtype=float),
            args=(station_coords, td),
            method="L-BFGS-B",
            options={"ftol": 1e-9, "disp": False},
        )

        if result.success:
            coords = result.x.tolist()
            logger.info("TDOA localized emitter at %s", coords)
            return coords

        logger.error("TDOA convergence failed: %s", result.message)
        return None
