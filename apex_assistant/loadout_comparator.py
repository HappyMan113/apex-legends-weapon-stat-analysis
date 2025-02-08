import itertools
import logging
import os
from functools import cmp_to_key
from statistics import correlation
from types import MappingProxyType
from typing import Generator, Mapping, Optional, Sequence, Tuple

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

    def compare_loadouts(self,
                         loadouts: Sequence[FullLoadout],
                         show_plots: bool = False) -> \
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
        cumulative_dmg_till_time_t: np.ndarray[np.floating] = np.empty((num_loadouts, len(ts)))
        for loadout_idx, loadout in enumerate(loadouts):
            check_type(FullLoadout, weapon=loadout)
            loadout: FullLoadout = loadout
            cumulative_dmg_till_time_t[loadout_idx] = loadout.get_cumulative_damage_vec(
                times_seconds=ts,
                distances_meters=ds)

        # You might think that earlier mean DPS values should be weighted more highly so that they
        # don't seem insignificant, but I think that if little damage is done, then they are in fact
        # insignificant. You might also think that longer encounters shouldn't be weighted highly,
        # but the weighting should already be taken into account for those via the lack of longer
        # encounters in the input data. Therefore, we should be simply looking at cumulative damage.
        expected_mean_dps = cumulative_dmg_till_time_t.mean(axis=1)
        sorti = expected_mean_dps.argsort()[::-1]

        sorted_loadouts_dict: dict[FullLoadout, float] = {
            loadouts[idx]: weighted_avg_damage
            for idx, weighted_avg_damage in zip(sorti, expected_mean_dps[sorti])
        }

        # sorti = np.empty(num_loadouts, dtype=int)
        # approximate_relative_scores = np.empty(num_loadouts, dtype=float)
        # approximate_relative_scores[0] = 1
        #
        # # We want earlier damage mean DPS values to be weighted more highly because otherwise short
        # # encounters will seem insignificant. However, we also want to avoid weighting them so
        # # highly that super short encounters approach infinite importance. To avoid the former
        # # pitfall, we need to scale everything by the max damage for any weapon for each time,
        # # and to avoid the latter pitfall, we need to scale the cumulative damage as opposed to the
        # # mean DPS. This normalization introduces another pitfall, which is that only the top
        # # result will be correct, as everything else is interfered with by the top result for each
        # # time. Therefore, the only meaningful result from a single comparison that can be gleaned
        # # is which loadout is best, and comparison of loadouts must be done iteratively.
        # #
        # # TODO: This still doesn't address the pitfall of loadouts lower down the rank interfering
        # #  with comparisons between loadouts higher in the rank due to the lower down one being
        # #  higher up at certain times. Maybe what I really need to do is sort using a comparator
        # #  that calculates the ratio between two weapons of cumulative damage at each time T and
        # #  then takes the mean of that and determines if that is greater than 1.
        # def comparator(cumulative_damage_till_time_t_for_loadout_a,
        #                cumulative_damage_till_time_t_for_loadout_b) -> int:
        #     cmp_result_vec = (cumulative_damage_till_time_t_for_loadout_a /
        #                       cumulative_damage_till_time_t_for_loadout_b)
        #
        #
        # sorted([], key=cmp_to_key(comparator))
        # cumulative_dmg_till_time_t_copy = cumulative_dmg_till_time_t.copy()
        # for loadout_idx in range(num_loadouts):
        #     max_damage_till_time_t = cumulative_dmg_till_time_t_copy.max(axis=0)
        #     if max_damage_till_time_t.all():
        #         normalized_cumulative_dmg_till_time_t = (cumulative_dmg_till_time_t_copy /
        #                                                  max_damage_till_time_t[np.newaxis, :])
        #     else:
        #         normalized_cumulative_dmg_till_time_t = cumulative_dmg_till_time_t_copy
        #     expected_mean_dps = normalized_cumulative_dmg_till_time_t.mean(axis=1)
        #
        #     best_loadout_indices = expected_mean_dps.argsort()
        #     best_loadout_index = best_loadout_indices[-1]
        #     next_loadout_idx = loadout_idx + 1
        #     if next_loadout_idx < num_loadouts:
        #         second_best_loadout_index = best_loadout_indices[-2]
        #         approximate_relative_score = min(
        #             expected_mean_dps[second_best_loadout_index] /
        #             expected_mean_dps[best_loadout_index],
        #             1)
        #         approximate_relative_scores[next_loadout_idx] = (
        #                 approximate_relative_score *
        #                 approximate_relative_scores[loadout_idx])
        #     sorti[loadout_idx] = best_loadout_index
        #
        #     # Zero out this one so that it is effectively not considered next iteration.
        #     cumulative_dmg_till_time_t_copy[best_loadout_index] = 0
        #
        # sorted_loadouts_dict: dict[FullLoadout, float] = {
        #     loadouts[idx]: weighted_avg_damage
        #     for idx, weighted_avg_damage in zip(sorti, approximate_relative_scores)
        # }

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
            ts_lin = np.linspace(ts.min(), ts.max() * 1.4, num=4000)
            t_sample_indices = np.abs(ts.reshape(-1, 1) - ts_lin.reshape(1, -1)).argmin(axis=1)
            for loadout in result.limit_to_best_num(10).get_sorted_loadouts():
                for distance_meters, config in loadout.get_distances_and_configs():
                    damages = loadout.get_cumulative_damage_with_config_vec(
                        config=config,
                        times_seconds=ts_lin,
                        distances_meters=np.full_like(ts_lin, fill_value=distance_meters))
                    damages *= 1 / ts_lin
                    ax2.plot(ts_lin, damages,
                             label=loadout.get_name(config),
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
