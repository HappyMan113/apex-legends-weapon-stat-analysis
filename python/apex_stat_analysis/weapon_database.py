import itertools
import logging
import os
from threading import Lock
from types import MappingProxyType
import typing

from matplotlib import pyplot as plt
import numpy as np

from apex_stat_analysis.weapon import WeaponArchetype, ConcreteWeapon, WeaponBase
from apex_stat_analysis.weapon_csv_parser import WeaponCsvReader, TTKCsvReader


logger = logging.getLogger()


class _ComparisonResult:
    def __init__(self,
                 sorted_archetypes: MappingProxyType[WeaponArchetype, tuple[float, WeaponBase]],
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
        if min_fraction_of_max <= 0:
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
        archetypes = self._get_best(self.sorted_archetypes, min_fraction_of_max=min_fraction_of_max)
        base_weapons = self._get_best(self.sorted_weapons, min_fraction_of_max=min_fraction_of_max)

        return _ComparisonResult(sorted_archetypes=archetypes,
                                 sorted_weapons=base_weapons)


class ApexDatabase:
    _INSTANCE: 'ApexDatabase | None' = None
    _LOCK = Lock()

    __create_key = object()

    def __init__(self, create_key):
        # TODO: Measure TTK in terms of duration of your active firing (i.e. not counting short
        #  pauses). Active firing means counting one round period per round fired. i.e. You can
        #  multiply number of rounds fired with round period and add reload time if you're in the
        #  open when reloading.
        assert create_key is ApexDatabase.__create_key, \
            (f'{self.__class__.__name__} objects must be created using {self.__class__.__name__}. '
             f'{self.get_instance.__name__}()')

        self_path = os.path.dirname(__file__)
        apex_stats_filename = os.path.join(self_path, 'Apex Legends Stats.csv')
        ttks_filename = os.path.join(self_path, 'Historic TTKs.csv')

        with open(apex_stats_filename, encoding='utf-8-sig') as fp:
            dr = WeaponCsvReader(fp)
            weapons: tuple[WeaponArchetype] = tuple(dr)

        with open(ttks_filename, encoding='utf-8-sig') as fp:
            dr = TTKCsvReader(fp)
            ttk_entries = tuple(dr)

        ttks = np.array(list(map(float, ttk_entries)))
        ttks.sort()
        self.ttks = ttks

        base_weapons: tuple[WeaponBase] = tuple(
            base_weapon
            for weapon_archetype in weapons
            for base_weapon in weapon_archetype.get_base_weapons())

        self._weapon_archetypes = weapons
        self._base_weapons = base_weapons

    def general_list(self, show_plots: bool = False):
        wingman: WeaponBase = next(
                base_weapon
                for base_weapon in self._base_weapons
                if 'wingman' in base_weapon.get_name().lower())
        base_weapons: list[WeaponBase] = [
            base_weapon.combine_with_sidearm(wingman)
            for base_weapon in self._base_weapons]
        for weapon_archetype in self._weapon_archetypes:
            try:
                base_weapons.extend(weapon_archetype.get_base_weapons(reload=True))
            except Exception as ex:
                raise type(ex)(f'trouble with getting base weapons for {weapon_archetype}') from ex

        result = self.compare_weapons(base_weapons).limit_to_best(0.75)

        sorted_weapons_str = '\n'.join(
                f'  {weighted_avg_damage:4.2f}: {weapon}'
                for weapon, weighted_avg_damage in
                result.get_sorted_weapons().items())
        print(f'Sorted Weapons:\n{sorted_weapons_str}')

        archetypes_str = '\n'.join(
                f'  {weapon}: {dmg:4.2f}'
                for weapon, (dmg, _) in
                result.get_archetypes().items())
        print(f'Sorted Weapon Archetypes:\n{archetypes_str}')

        if show_plots:
            ttks = self.ttks
            ts_lin = np.linspace(ttks.min(), ttks.max(), num=1000)
            for base_weapon in result.get_sorted_weapons():
                damages = np.array([base_weapon.get_cumulative_damage(t) for t in ts_lin])
                damages *= 1 / ts_lin
                plt.plot(ts_lin, damages, label=base_weapon.get_name())

            plt.ylim((0, None))
            plt.legend()
            plt.show()

    def compare_weapons(self, base_weapons: typing.Iterable[WeaponBase]) -> _ComparisonResult:
        if not isinstance(base_weapons, (tuple, list)):
            base_weapons = tuple(base_weapons)

        ts = self.ttks
        power = 0

        # ts = np.linspace(self.ttks.min(), self.ttks.max(), num=1000)
        # # power should be non-positive, probably in the range [0, -1]. Lower values means you're
        # # assuming TTK is going to be lower. Zero probably makes sense if your TTK values are
        # # good.
        # power = -1

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
                                             weights=(ts ** power))
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

    @staticmethod
    def get_instance() -> 'ApexDatabase':
        with ApexDatabase._LOCK:
            if ApexDatabase._INSTANCE is None:
                ApexDatabase._INSTANCE = ApexDatabase(ApexDatabase.__create_key)

        return ApexDatabase._INSTANCE

    def get_weapon_archetypes(self) -> tuple[WeaponArchetype]:
        return self._weapon_archetypes

    def get_base_weapons(self):
        return self._base_weapons

    # def get_base_weapon(self, main_weapon: ApexTermBase, sidearm: ApexTermBase | None = None) -> \
    #         WeaponBase | None:
    #     if sidearm is not None:
    #         reload = True
    #     else:
    #         reload = False
    #
    #     self.get_weapon_archetype()
    #     return
