import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import weibull_min as weibull


def _get_readonly_view(arr: NDArray) -> NDArray:
    arr = arr.view()
    arr.flags.writeable = False
    return arr


class SupportedDistanceMeters(IntEnum):
    FIVE = 5
    TEN = 10
    TWENTY = 20
    FORTY = 40
    EIGHTY = 80
    ONE_HUNDRED_AND_SIXTY = 160


HIPFIRE_DISTANCE_METERS: float = 15


# Player accuracy. This is a fraction that gets multiplied by the "ideal" accuracy value.
PLAYER_ACCURACY: float = 0.75

# We are going to permute all possible shot hit permutations for a certain number of initial
# weapons, and calculate the probability of each permutation.
MAX_NUM_PERMUTED_WEAPONS: int = 5

# Damage to kill to assume when determining P(kill). This would be how much health + shields an
# enemy has in a 1v1, or how much total health multiple enemies have in a 1v2 or 1v3 scenario.
__DAMAGES_TO_KILL = _get_readonly_view(np.array([175, 200, 215], dtype=float))
_DAMAGES_TO_KILL_WEIGHTS = _get_readonly_view(np.array([0.25, 1, 0.5], dtype=float))

# Distances to assume when determining P(kill).
__DISTANCES_METERS: Tuple[SupportedDistanceMeters, ...] = (
    SupportedDistanceMeters.FIVE,
    SupportedDistanceMeters.TEN,
    SupportedDistanceMeters.TWENTY,
    SupportedDistanceMeters.FORTY,
    SupportedDistanceMeters.EIGHTY,
    SupportedDistanceMeters.ONE_HUNDRED_AND_SIXTY
)
_DISTANCES_METERS_WEIGHTS = _get_readonly_view(np.array([1, 1, 1, 2, 2, 2], dtype=float))

# k should be greater than 1, because the longer you stay in a fight, the less health you tend to
# have, and therefore the more likely you are to get finished off at any given moment. I don't
# exactly have the raw statistics from which to estimate k, but... you know... 10 is probably too
# high based on anecdotal experience. Just reason with me here for a minute. If k is 1,
# that would mean your chances of death at any given time are constant. If k is 2, that would
# mean your chances of death at any given time increase linearly over time. If k is 3, that would
# mean your chances of death at any given time increase quadratically over time. What we can say is
# that as an encounter drags on, opponents are more likely to have gotten their bearings and started
# dealing more serious damage, and their teammates are more likely to start joining in,
# meaning that incoming damage increases over time. However, the shape of the curve describing
# incoming damage over time is not immediately obvious. If we assume linearly increasing damage
# over time, that would mean quadratically decreasing health over time. The parameters below are
# what I came up with when trying to fit the Weibull survival function against my own ver y
# k=1 - no way that's right; the peak of the density is at 0
# k=1.25 - peaks really early, but maybe that's believable
# k=1.5 - believable if shifted to the right
# k=2 - somewhat believable
# k=2.5 - dubious
# k>=3 - no way that's right; it doesn't allow for drawn-out encounters
__K: float = 1.25
__LMBDA: float = 2  # Should probably be somewhere between 1 & 3.

# It is technically possible to get a die faster than 0.5 seconds, but assuming that this is the
# minimum results in a more believable probability density.
min_time_to_kill: float = 0.5
__NUM_TIME_SAMPLES: int = 10

__percentiles = np.linspace(0, 1, num=__NUM_TIME_SAMPLES, endpoint=False)
__TIMES_AVAILABLE_TO_KILL = weibull.ppf(__percentiles, c=__K, scale=__LMBDA) + min_time_to_kill


# noinspection PyUnusedLocal
def get_distance_time_multiplier(distance_meters: SupportedDistanceMeters) -> float:
    return 1

# Fixed time windows to assume when determining how much time we have to accumulate P(kill).
_TIMES_AVAILABLE_TO_KILL_WEIGHTS = np.ones_like(__TIMES_AVAILABLE_TO_KILL, dtype=float)


sorti = __DAMAGES_TO_KILL.argsort()
__DAMAGES_TO_KILL = _get_readonly_view(__DAMAGES_TO_KILL[sorti])
_DAMAGES_TO_KILL_WEIGHTS = _get_readonly_view(_DAMAGES_TO_KILL_WEIGHTS[sorti] /
                                              _DAMAGES_TO_KILL_WEIGHTS.sum())

filteri = np.flatnonzero(_DISTANCES_METERS_WEIGHTS > 0)
__DISTANCES_METERS = tuple(__DISTANCES_METERS[idx] for idx in filteri)
_DISTANCES_METERS_WEIGHTS = _DISTANCES_METERS_WEIGHTS[filteri]
sorti = np.argsort(__DISTANCES_METERS)
__DISTANCES_METERS = tuple(__DISTANCES_METERS[idx] for idx in sorti)
_DISTANCES_METERS_WEIGHTS = _get_readonly_view(_DISTANCES_METERS_WEIGHTS[sorti] /
                                               _DISTANCES_METERS_WEIGHTS.sum())

sorti = __TIMES_AVAILABLE_TO_KILL.argsort()
__TIMES_AVAILABLE_TO_KILL = _get_readonly_view(__TIMES_AVAILABLE_TO_KILL[sorti])
_TIMES_AVAILABLE_TO_KILL_WEIGHTS = _get_readonly_view(_TIMES_AVAILABLE_TO_KILL_WEIGHTS[sorti] /
                                                      _TIMES_AVAILABLE_TO_KILL_WEIGHTS.sum())

if __DAMAGES_TO_KILL.shape != _DAMAGES_TO_KILL_WEIGHTS.shape:
    raise ValueError('len(__DAMAGES_TO_KILL) != len(__DAMAGES_TO_KILL_WEIGHTS)')
if (len(__DISTANCES_METERS),) != _DISTANCES_METERS_WEIGHTS.shape:
    raise ValueError('len(__DISTANCES_METERS) != len(__DISTANCES_METERS_WEIGHTS)')
if __TIMES_AVAILABLE_TO_KILL.shape != _TIMES_AVAILABLE_TO_KILL_WEIGHTS.shape:
    raise ValueError('len(__TIMES_AVAILABLE_TO_KILL) != len(__TIMES_AVAILABLE_TO_KILL_WEIGHTS)')

assert math.isclose(_DAMAGES_TO_KILL_WEIGHTS.sum(), 1)
assert math.isclose(_DISTANCES_METERS_WEIGHTS.sum(), 1)
assert math.isclose(_TIMES_AVAILABLE_TO_KILL_WEIGHTS.sum(), 1)


def get_p_kill_for_damages_to_kill(p_kill_array: Iterable[float]) -> float:
    if not isinstance(p_kill_array, np.ndarray):
        p_kill_array = np.fromiter(p_kill_array,
                                   dtype=float,
                                   count=len(_DAMAGES_TO_KILL_WEIGHTS))
    if p_kill_array.shape != _DAMAGES_TO_KILL_WEIGHTS.shape:
        raise ValueError('P(kill) array must have one dimension!')
    return float((p_kill_array * _DAMAGES_TO_KILL_WEIGHTS).sum())


def get_p_kill_for_distances(p_kill_array: Iterable[float]) -> float:
    if not isinstance(p_kill_array, np.ndarray):
        p_kill_array = np.fromiter(p_kill_array,
                                   dtype=float,
                                   count=len(_DISTANCES_METERS_WEIGHTS))
    if p_kill_array.shape != _DISTANCES_METERS_WEIGHTS.shape:
        raise ValueError('P(kill) array must have one dimension!')
    return float((p_kill_array * _DISTANCES_METERS_WEIGHTS).sum())


def get_p_kill_for_times(p_kill_array: Iterable[float]) -> float:
    if not isinstance(p_kill_array, np.ndarray):
        p_kill_array = np.fromiter(p_kill_array,
                                   dtype=float,
                                   count=len(_TIMES_AVAILABLE_TO_KILL_WEIGHTS))
    if p_kill_array.shape != _TIMES_AVAILABLE_TO_KILL_WEIGHTS.shape:
        raise ValueError('P(kill) array must have one dimension!')
    return float((p_kill_array * _TIMES_AVAILABLE_TO_KILL_WEIGHTS).sum())


@dataclass(frozen=True)
class Config:
    damages_to_kill: NDArray[np.float64]
    distances_meters: Tuple[SupportedDistanceMeters, ...]
    times_given_seconds: NDArray[np.float64]


CONFIG = Config(damages_to_kill=__DAMAGES_TO_KILL,
                distances_meters=__DISTANCES_METERS,
                times_given_seconds=__TIMES_AVAILABLE_TO_KILL)
