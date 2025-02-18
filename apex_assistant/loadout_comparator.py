import itertools
import logging
import os
from types import MappingProxyType
from typing import Dict, Generator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from apex_assistant.checker import check_float, check_float_vec, check_int, check_tuple
from apex_assistant.weapon import FullLoadout, Weapon

logger = logging.getLogger()


class ComparisonResult:
    def __init__(self, sorted_loadouts: Mapping[FullLoadout, float]):
        self.sorted_loadouts = MappingProxyType(sorted_loadouts)
        self.weighted_average_damage = np.array(list(sorted_loadouts.values()))

    def _calc_min(self, min_fraction_of_max: float) -> float:
        min_dmg = self.weighted_average_damage.max() * min_fraction_of_max
        return min_dmg

    def _get_best(self,
                  weapon_dict: MappingProxyType,
                  min_fraction_of_max: float = 0) -> MappingProxyType:
        min_dmg = self._calc_min(min_fraction_of_max)
        num = np.count_nonzero(self.weighted_average_damage >= min_dmg)
        return self._get_best_num(weapon_dict, num)

    @staticmethod
    def _get_best_num(weapon_dict: MappingProxyType, num: int) -> MappingProxyType:
        assert num >= 1
        if num >= len(weapon_dict):
            return weapon_dict

        return MappingProxyType({
            weapon: tup
            for weapon, tup in itertools.islice(weapon_dict.items(), num)})

    def __len__(self):
        return len(self.sorted_loadouts)

    def get_sorted_loadouts(self) -> MappingProxyType[FullLoadout, float]:
        return self.sorted_loadouts

    def get_best_loadout(self) -> tuple[FullLoadout, float]:
        return self.get_nth_best_loadout(1)

    def get_nth_best_loadout(self, n_one_indexed: int) -> tuple[FullLoadout, float]:
        assert n_one_indexed >= 1
        n_one_indexed = min(n_one_indexed, len(self.sorted_loadouts))
        return list(itertools.islice(self.sorted_loadouts.items(), n_one_indexed))[-1]

    def limit_to_best(self, min_fraction_of_max: float = 0):
        assert min_fraction_of_max <= 1
        base_weapons = self._get_best(self.sorted_loadouts, min_fraction_of_max=min_fraction_of_max)

        return ComparisonResult(sorted_loadouts=base_weapons)

    def limit_to_best_num(self, num: int) -> 'ComparisonResult':
        check_int(num=num, min_value=1)
        base_weapons = self._get_best_num(self.sorted_loadouts, num=num)

        return ComparisonResult(sorted_loadouts=base_weapons)

    def __repr__(self) -> str:
        delim = '\n'
        return delim + delim.join(
            f'  {weighted_avg_damage:6.2f}: {weapon}'
            for weapon, weighted_avg_damage in self.get_sorted_loadouts().items())


class LoadoutComparator:
    def __init__(self,
                 damages_to_kill: NDArray[np.float64],
                 damages_to_kill_weights: NDArray[np.float64],
                 distances_meters: NDArray[np.float64],
                 distances_meters_weights: NDArray[np.float64],
                 player_accuracy: float):
        check_float_vec(damages_to_kill=damages_to_kill,
                        damages_to_kill_weights=damages_to_kill_weights,
                        distances_meters=distances_meters,
                        distances_meters_weights=distances_meters_weights)
        check_float(player_accuracy=player_accuracy,
                    min_value=0,
                    min_is_exclusive=True,
                    max_value=1)

        sorti = damages_to_kill.argsort()
        damages_to_kill = damages_to_kill[sorti]
        damages_to_kill_weights = damages_to_kill_weights[sorti] / damages_to_kill_weights.sum()

        sorti = distances_meters.argsort()
        distances_meters = distances_meters[sorti]
        distances_meters_weights = distances_meters_weights[sorti] / distances_meters_weights.sum()

        self._damages_to_kill = damages_to_kill
        self._damages_to_kill_weights = damages_to_kill_weights
        self._distances_meters = distances_meters
        self._distances_meters_weights = distances_meters_weights
        self._player_accuracy = player_accuracy

    def _get_mean_dps_for_distances(self, dps_array: NDArray[np.float64]) -> NDArray[np.float64]:
        if dps_array.ndim != 2:
            raise ValueError('DPS array must have TWO dimensions!')
        return (dps_array * self._damages_to_kill_weights[:, np.newaxis]).sum(axis=0)

    def _get_mean_dps_for_damages(self, mean_dps_for_distances: NDArray[np.float64]) -> float:
        if mean_dps_for_distances.ndim != 1:
            raise ValueError('mean_dps_for_distances must have ONE dimension!')
        return (mean_dps_for_distances * self._distances_meters_weights).sum()

    def _get_mean_dps(self, dps_array: NDArray[np.float64]) -> float:
        if dps_array.ndim != 2:
            raise ValueError('DPS array must have TWO dimensions!')
        return self._get_mean_dps_for_damages(self._get_mean_dps_for_distances(dps_array))

    def compare_loadouts(self, loadouts: Sequence[FullLoadout], show_plots: bool = False) -> \
            ComparisonResult:
        if len(loadouts) == 0:
            raise ValueError('loadouts cannot be empty!')

        damages_to_kill = self._damages_to_kill
        distances_meters = self._distances_meters
        player_accuracy = self._player_accuracy

        # Loadouts in descending order
        loadouts_and_dpss: List[Tuple[FullLoadout, NDArray[np.float64]]] = sorted(
            ((loadout, loadout.get_dps_vec(damages_to_kill=damages_to_kill,
                                           distances_meters=distances_meters,
                                           player_accuracy=player_accuracy))
             for loadout in frozenset(loadouts)),
            key=lambda loadout_and_dps_vec: self._get_mean_dps(loadout_and_dps_vec[1]),
            reverse=True)

        sorted_loadouts_dict: Dict[FullLoadout, float] = {loadout: self._get_mean_dps(dpss)
                                                          for loadout, dpss in loadouts_and_dpss}

        result = ComparisonResult(sorted_loadouts_dict)

        if show_plots:
            sorted_loadouts_str = str(result)
            if sorted_loadouts_str.count('\n') > 100:
                filename = os.path.abspath('comparison_results.log')
                with open(filename, 'w+') as fp:
                    fp.write(sorted_loadouts_str)
                logger.info(f'Wrote result to {filename}')
            else:
                logger.info(f'Sorted Loadouts: {sorted_loadouts_str}')

            from matplotlib import pyplot as plt

            _, (ax1, ax2) = plt.subplots(ncols=2)

            ts_min = 0
            ts_max = 2
            ts_lin = np.linspace(ts_min, ts_max, num=4000)

            for loadout, dpss in loadouts_and_dpss[:10]:
                mean_dps_for_distances = self._get_mean_dps_for_distances(dpss)
                ax1.plot(distances_meters, mean_dps_for_distances, label=loadout.get_name())

                damages = loadout.get_cumulative_damage_vec(
                    times_seconds=ts_lin,
                    distances_meters=np.zeros_like(ts_lin),
                    player_accuracy=player_accuracy)
                ax2.plot(ts_lin, damages, label=loadout.get_name())

            ax1.set_xlabel('Distance (meters)')
            ax1.set_ylabel('Expected Mean DPS')
            ax1.legend()

            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('DPS (assuming perfect accuracy)')
            ax2.set_ylim((0, None))
            ax2.legend()

            plt.show()

        return result

    def get_best_loadouts(self,
                          weapons: Tuple[Weapon, ...],
                          max_num_loadouts: Optional[int] = None) -> \
            Generator[FullLoadout, None, None]:
        check_tuple(Weapon, allow_empty=False, weapons=weapons)
        check_int(min_value=1, optional=True, max_num_loadouts=max_num_loadouts)

        weapons_set = frozenset(weapons)
        exclude_filter: set[Weapon] = set()
        max_max_num_loadouts = len(weapons_set) - FullLoadout.NUM_WEAPONS + 1
        if max_num_loadouts is None:
            max_num_loadouts = max_max_num_loadouts
        else:
            max_num_loadouts = min(max_max_num_loadouts, max_num_loadouts)

        loadouts = tuple(FullLoadout.get_loadouts(weapons_set))
        comparison_result = self.compare_loadouts(loadouts)

        num_loadouts = 0
        for best_loadout in comparison_result.get_sorted_loadouts():
            if not best_loadout.get_weapons().isdisjoint(exclude_filter):
                continue

            exclude_filter.add(best_loadout.get_weapon_a())

            yield best_loadout
            num_loadouts += 1
            if num_loadouts >= max_num_loadouts:
                break
