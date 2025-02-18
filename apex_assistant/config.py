# Arbitrary.
import math

import numpy as np

# We are going to permute all possible shot hit permutations for a certain number of initial
# shots, and calculate the probability of each permutation.
MAX_NUM_PERMUTED_ROUNDS = 5

DAMAGES_TO_KILL = np.array([175, 200, 215], dtype=float)
DAMAGES_TO_KILL_WEIGHTS = np.array([0.25, 1, 0.5] * len(DAMAGES_TO_KILL), dtype=float)
DISTANCES_METERS = np.array([10, 20, 40, 80, 160], dtype=float)
DISTANCES_METERS_WEIGHTS = np.array([1, 1, 1, 1, 1], dtype=float)
PLAYER_ACCURACY: float = 0.75

CONFIG_FIGURING_DAMAGE_TO_KILL: float = DAMAGES_TO_KILL.max()
MAX_CONFIG_FIGURING_DISTANCE_METERS: int = math.ceil(DISTANCES_METERS.max())
