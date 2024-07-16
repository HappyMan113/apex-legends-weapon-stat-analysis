import itertools
import logging
from types import MappingProxyType
from typing import Iterable, Tuple

import numpy as np

from apex_assistant.checker import check_int, check_tuple
from apex_assistant.ttk_datum import TTKDatum
from apex_assistant.weapon import ConcreteWeapon, WeaponArchetype, WeaponBase

logger = logging.getLogger()


class _ComparisonResult:
    def __init__(self,
                 sorted_archetypes: MappingProxyType[WeaponArchetype, Tuple[float, WeaponBase]],
                 sorted_weapons: MappingProxyType[WeaponBase, float]):
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

    def get_archetypes(self) -> MappingProxyType[WeaponArchetype, tuple[float, WeaponBase]]:
        return self.sorted_archetypes

    def get_sorted_weapons(self) -> MappingProxyType[WeaponBase, float]:
        return self.sorted_weapons

    def get_best_weapon(self) -> tuple[WeaponBase, float]:
        return self.get_nth_best_weapon(1)

    def get_nth_best_weapon(self, n_one_indexed: int) -> tuple[WeaponBase, float]:
        assert n_one_indexed >= 1
        n_one_indexed = min(n_one_indexed, len(self.sorted_weapons))
        return list(itertools.islice(self.sorted_weapons.items(), n_one_indexed))[-1]

    def limit_to_best(self, min_fraction_of_max: float = 0):
        assert min_fraction_of_max <= 1
        archetypes = self._get_best(self.sorted_archetypes, min_fraction_of_max=min_fraction_of_max)
        base_weapons = self._get_best(self.sorted_weapons, min_fraction_of_max=min_fraction_of_max)

        return _ComparisonResult(sorted_archetypes=archetypes,
                                 sorted_weapons=base_weapons)

    def limit_to_best_num(self, num: int) -> '_ComparisonResult':
        check_int(num=num, min_value=1)
        archetypes = self._get_best_num(self.sorted_archetypes, num=num)
        base_weapons = self._get_best_num(self.sorted_weapons, num=num)

        return _ComparisonResult(sorted_archetypes=archetypes,
                                 sorted_weapons=base_weapons)

    def __repr__(self) -> str:
        delim = '\n'
        return delim + delim.join(
            f'  {weighted_avg_damage:4.2f}: {weapon}'
            for weapon, weighted_avg_damage in self.get_sorted_weapons().items())


class WeaponComparer:
    def __init__(self, ttk_entries: tuple[TTKDatum, ...]):
        check_tuple(TTKDatum, ttk_entries=ttk_entries)

        ttks = np.array(list(map(float, ttk_entries)))
        ttks.sort()
        self.ttks = ttks

    def compare_weapons(self, base_weapons: Iterable[WeaponBase]) -> _ComparisonResult:
        if not isinstance(base_weapons, tuple):
            base_weapons = tuple(base_weapons)

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

        num_base_weapons = len(base_weapons)
        damage_table: np.ndarray[np.floating] = np.empty((num_base_weapons, len(ts)))
        for idx, base_weapon in enumerate(base_weapons):
            base_weapon: ConcreteWeapon = base_weapon

            # This could be made faster by using vectorized operations.
            damages_cum = np.array([base_weapon.get_cumulative_damage(t) for t in ts])
            damage_table[idx] = damages_cum

        # This is the mean dps up till time t for t in TTKs.
        mean_dps_till_time_t = damage_table * (1 / ts)

        # This one actually weights earlier damage more highly.
        weighted_average_damage = np.average(mean_dps_till_time_t,
                                             axis=1,
                                             weights=weights)
        sorti = weighted_average_damage.argsort()[::-1]
        sorted_weapons: tuple[WeaponBase, ...] = tuple(base_weapons[idx] for idx in sorti)

        sorted_weapons_dict: dict[WeaponBase, float] = {
            weapon: weighted_avg_damage
            for weapon, weighted_avg_damage in zip(sorted_weapons, weighted_average_damage[sorti])
        }
        if len(sorted_weapons_dict) < num_base_weapons:
            logger.warning('Duplicate weapons found. Only unique weapons will be compared.')

        sorted_archetypes_dict: dict[WeaponArchetype, tuple[float, WeaponBase]] = {}
        for weapon, weighted_avg_damage in sorted_weapons_dict.items():
            if weapon.get_archetype() not in sorted_archetypes_dict:
                sorted_archetypes_dict[weapon.get_archetype()] = (weighted_avg_damage, weapon)

        return _ComparisonResult(sorted_archetypes=MappingProxyType(sorted_archetypes_dict),
                                 sorted_weapons=MappingProxyType(sorted_weapons_dict))

    def general_list(self,
                     base_weapons: tuple[ConcreteWeapon, ...],
                     sidearm: ConcreteWeapon | None = None,
                     show_plots: bool = False):
        without_wingman_weapons = tuple(weapon.reload() for weapon in base_weapons)
        if sidearm is not None:
            with_wingman_weapons = tuple(weapon.combine_with_sidearm(sidearm).reload()
                                         for weapon in base_weapons)
            base_weapons: tuple[WeaponBase, ...] = without_wingman_weapons + with_wingman_weapons
        else:
            base_weapons: tuple[WeaponBase, ...] = without_wingman_weapons

        result = self.compare_weapons(base_weapons)

        sorted_weapons_str = '\n'.join(
            f'  {weighted_avg_damage:4.2f}: {weapon}'
            for weapon, weighted_avg_damage in
            result.get_sorted_weapons().items())
        logger.info(f'Sorted Weapons:\n{sorted_weapons_str}')

        archetypes_str = '\n'.join(
            f'  {base_weapon}: {dmg:4.2f}'
            for weapon, (dmg, base_weapon) in
            result.get_archetypes().items())
        logger.info(f'Sorted Weapon Archetypes:\n{archetypes_str}')

        if show_plots:
            result = result.limit_to_best_num(4)
            from matplotlib import pyplot as plt

            ttks = self.ttks
            ts_lin = np.linspace(ttks.min(), ttks.max(), num=1000)
            for base_weapon in result.get_sorted_weapons():
                damages = np.array([base_weapon.get_cumulative_damage(t) for t in ts_lin])
                damages *= 1 / ts_lin
                plt.plot(ts_lin, damages, label=base_weapon.get_name())

            plt.ylim((0, None))
            plt.legend()
            plt.show()
