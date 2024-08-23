import itertools
import logging
import os
from statistics import correlation
from types import MappingProxyType
from typing import Generator, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from apex_assistant.checker import check_int, check_tuple, check_type
from apex_assistant.ttk_entry import Engagement
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
    def __init__(self, engagements: Tuple[Engagement, ...]):
        check_tuple(Engagement, engagements=engagements)
        engagements = sorted(engagements, key=lambda e: e.get_ttff_seconds())

        self._ttffs: np.ndarray[np.floating] = np.array([ttk.get_ttff_seconds()
                                                         for ttk in engagements])
        self._enemy_distances_meters: np.ndarray[np.floating] = np.array([
            ttk.get_enemy_distance_meters()
            for ttk in engagements])

    def compare_loadouts(self, loadouts: Sequence[FullLoadout], show_plots: bool = False) -> \
            ComparisonResult:
        if len(loadouts) == 0:
            raise ValueError('loadouts cannot be empty!')

        loadout_set = frozenset(loadouts)
        if len(loadout_set) < len(loadouts):
            logger.debug('Duplicate loadouts found. Only unique loadouts will be compared.')
            loadouts = tuple(loadout_set)

        ts = self._ttffs
        ds = self._enemy_distances_meters

        num_loadouts = len(loadouts)
        damage_table: np.ndarray[np.floating] = np.empty((num_loadouts, len(ts)))
        for idx, loadout in enumerate(loadouts):
            check_type(FullLoadout, weapon=loadout)
            loadout: FullLoadout = loadout
            damage_table[idx] = np.array([
                loadout.get_cumulative_damage(time_seconds=time_seconds,
                                              distance_meters=distance_meters)
                for time_seconds, distance_meters in zip(ts, ds)])

        # This is the mean dps up till time t for t in TTKs.
        mean_dps_till_time_t = damage_table * (1 / ts)

        # This one actually weights earlier damage more highly.
        expected_mean_dps = mean_dps_till_time_t.mean(axis=1)
        sorti = expected_mean_dps.argsort()[::-1]

        sorted_loadouts_dict: dict[FullLoadout, float] = {
            loadouts[idx]: weighted_avg_damage
            for idx, weighted_avg_damage in zip(sorti, expected_mean_dps[sorti])
        }

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

            ax1.scatter(ds, ts)
            ax1.set_xlabel('Distance (meters)')
            ax1.set_ylabel('Time to Finish Firing (seconds)')
            ax1.set_xlim((0, None))
            ax1.set_ylim((0, None))
            corr = correlation(ds, ts)
            ax1.set_title(f'Correlation: {corr}')

            ax2.axvline(ts.min())
            ax2.axvline(ts.max())
            ts_lin = np.linspace(0.4, ts.max() * 1.4, num=4000)
            t_sample_indices = np.abs(ts.reshape(-1, 1) - ts_lin.reshape(1, -1)).argmin(axis=1)
            for loadout in result.limit_to_best_num(10).get_sorted_loadouts():
                damages = np.array([
                    loadout.get_cumulative_damage(time_seconds=time_seconds,
                                                  distance_meters=0)
                    for time_seconds in ts_lin])
                damages *= 1 / ts_lin
                ax2.plot(ts_lin, damages,
                         label=loadout.get_name(),
                         markevery=t_sample_indices,
                         markeredgecolor='red',
                         marker='x')

            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Expected Mean DPS')
            ax2.set_ylim((0, None))

            plt.legend()
            plt.show()

        return result

    def get_best_loadouts(self,
                          weapons: Tuple[Weapon],
                          max_num_loadouts: Optional[int] = None) -> \
            Generator[FullLoadout, None, None]:
        check_tuple(Weapon, allow_empty=False, weapons=weapons)
        check_int(min_value=1, optional=True, max_num_loadouts=max_num_loadouts)

        weapons_set = set(weapons)
        max_max_num_loadouts = len(weapons_set) - FullLoadout.NUM_WEAPONS + 1
        if max_num_loadouts is None:
            max_num_loadouts = max_max_num_loadouts
        else:
            max_num_loadouts = min(max_max_num_loadouts, max_num_loadouts)

        num_loadouts = 0
        while num_loadouts < max_num_loadouts:
            best_loadout = self._compare_loadouts_containing_only(weapons_set)
            weapons_set.remove(best_loadout.get_weapon_a())

            yield best_loadout
            num_loadouts += 1

    def _compare_loadouts_containing_only(self, weapons: Iterable[Weapon]) -> FullLoadout:
        loadouts = tuple(FullLoadout.get_loadouts(weapons))
        best_loadout, _ = self.compare_loadouts(loadouts).get_best_loadout()
        return best_loadout
