import itertools
import logging
import os
from types import MappingProxyType
from typing import Generator, Iterable, Optional, Sequence, Tuple

import numpy as np

from apex_assistant.checker import check_int, check_tuple, check_type
from apex_assistant.weapon import FullLoadout, Weapon, WeaponArchetype

logger = logging.getLogger()


class ComparisonResult:
    def __init__(self,
                 sorted_archetypes: MappingProxyType[WeaponArchetype, Tuple[float, FullLoadout]],
                 sorted_weapons: MappingProxyType[FullLoadout, float]):
        self.sorted_archetypes = sorted_archetypes
        self.sorted_weapons = sorted_weapons
        self.weighted_average_damage = np.array(list(sorted_weapons.values()))

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

    def get_archetypes(self) -> MappingProxyType[WeaponArchetype, tuple[float, FullLoadout]]:
        return self.sorted_archetypes

    def get_sorted_loadouts(self) -> MappingProxyType[FullLoadout, float]:
        return self.sorted_weapons

    def get_best_loadout(self) -> tuple[FullLoadout, float]:
        return self.get_nth_best_loadout(1)

    def get_nth_best_loadout(self, n_one_indexed: int) -> tuple[FullLoadout, float]:
        assert n_one_indexed >= 1
        n_one_indexed = min(n_one_indexed, len(self.sorted_weapons))
        return list(itertools.islice(self.sorted_weapons.items(), n_one_indexed))[-1]

    def limit_to_best(self, min_fraction_of_max: float = 0):
        assert min_fraction_of_max <= 1
        archetypes = self._get_best(self.sorted_archetypes, min_fraction_of_max=min_fraction_of_max)
        base_weapons = self._get_best(self.sorted_weapons, min_fraction_of_max=min_fraction_of_max)

        return ComparisonResult(sorted_archetypes=archetypes,
                                sorted_weapons=base_weapons)

    def limit_to_best_num(self, num: int) -> 'ComparisonResult':
        check_int(num=num, min_value=1)
        archetypes = self._get_best_num(self.sorted_archetypes, num=num)
        base_weapons = self._get_best_num(self.sorted_weapons, num=num)

        return ComparisonResult(sorted_archetypes=archetypes,
                                sorted_weapons=base_weapons)

    def __repr__(self) -> str:
        delim = '\n'
        return delim + delim.join(
            f'  {weighted_avg_damage:6.2f}: {weapon}'
            for weapon, weighted_avg_damage in self.get_sorted_loadouts().items())


class LoadoutComparator:
    def __init__(self, ttks: Tuple[float, ...]):
        check_tuple((float, int), ttks=ttks)

        ttks = np.array(ttks)
        ttks.sort()
        self.ttks = ttks

    def compare_loadouts(self, loadouts: Sequence[FullLoadout], show_plots: bool = False) -> \
            ComparisonResult:
        if len(loadouts) == 0:
            raise ValueError('loadouts cannot be empty!')
        tiebreaker_time = 10

        ts = self.ttks
        weights = np.ones_like(ts)

        # ts = np.linspace(self.ttks.min(), self.ttks.max(), num=1000)
        # # power should be non-positive, probably in the range [0, -1]. Lower values means you're
        # # assuming TTK is going to be lower. Zero probably makes sense if your TTK values are
        # # good.
        # power = -1
        # weights = ts ** power

        if ts.max() < tiebreaker_time:
            ts = np.append(ts, tiebreaker_time)
            weights = np.append(weights, weights.sum() * 0.01)

        num_base_weapons = len(loadouts)
        damage_table: np.ndarray[np.floating] = np.empty((num_base_weapons, len(ts)))
        for idx, weapon in enumerate(loadouts):
            check_type(FullLoadout, weapon=weapon)
            weapon: FullLoadout = weapon
            damage_table[idx] = np.array([weapon.get_cumulative_damage(t) for t in ts])

        # This is the mean dps up till time t for t in TTKs.
        mean_dps_till_time_t = damage_table * (1 / ts)

        # This one actually weights earlier damage more highly.
        expected_mean_dps = np.average(mean_dps_till_time_t, axis=1, weights=weights)
        sorti = expected_mean_dps.argsort()[::-1]

        sorted_weapons_dict: dict[FullLoadout, float] = {
            loadouts[idx]: weighted_avg_damage
            for idx, weighted_avg_damage in zip(sorti, expected_mean_dps[sorti])
        }
        if len(sorted_weapons_dict) < num_base_weapons:
            logger.warning('Duplicate weapons found. Only unique weapons will be compared.')

        sorted_archetypes_dict: dict[WeaponArchetype, tuple[float, FullLoadout]] = {}
        for loadout, weighted_avg_damage in sorted_weapons_dict.items():
            if loadout.get_main_loadout().get_archetype() not in sorted_archetypes_dict:
                sorted_archetypes_dict[loadout.get_main_loadout().get_archetype()] = (
                    weighted_avg_damage,
                    loadout)

        result = ComparisonResult(sorted_archetypes=MappingProxyType(sorted_archetypes_dict),
                                  sorted_weapons=MappingProxyType(sorted_weapons_dict))

        if show_plots:
            sorted_loadouts_str = str(result)
            if sorted_loadouts_str.count('\n') > 100:
                filename = os.path.abspath('comparison_results.log')
                with open(filename, 'w+') as fp:
                    fp.write(sorted_loadouts_str)
                logger.info(f'Wrote result to {filename}')
            else:
                logger.info(f'Sorted Loadouts: {sorted_loadouts_str}')

            archetypes_str = '\n'.join(
                f'  {weighted_avg_damage:6.2f}: {base_weapon}'
                for weapon, (weighted_avg_damage, base_weapon) in
                result.get_archetypes().items())
            logger.info(f'Sorted Weapon Archetypes:\n{archetypes_str}')

            from matplotlib import pyplot as plt

            plt.axvline(ts.min())
            plt.axvline(ts.max())
            ts_lin = np.linspace(0.4, ts.max() * 1.4, num=4000)
            t_sample_indices = np.abs(ts.reshape(-1, 1) - ts_lin.reshape(1, -1)).argmin(axis=1)
            for base_weapon in result.limit_to_best_num(10).get_sorted_loadouts():
                damages = np.array([base_weapon.get_cumulative_damage(t) for t in ts_lin])
                damages *= 1 / ts_lin
                plt.plot(ts_lin, damages, label=base_weapon.get_name(),
                         markevery=t_sample_indices,
                         markeredgecolor='red',
                         marker='x')

            plt.ylim((0, None))
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
            weapons_set.remove(best_loadout.get_main_weapon())

            yield best_loadout
            num_loadouts += 1

    def _compare_loadouts_containing_only(self, weapons: Iterable[Weapon]) -> FullLoadout:
        loadouts = tuple(FullLoadout.get_loadouts(weapons))
        best_loadout, _ = self.compare_loadouts(loadouts).get_best_loadout()
        return best_loadout
