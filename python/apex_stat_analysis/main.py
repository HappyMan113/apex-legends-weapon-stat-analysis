import os.path

from matplotlib import pyplot as plt
import numpy as np

from apex_stat_analysis.weapon import ConcreteWeapon, WeaponArchetype, WeaponBase
from apex_stat_analysis.weapon_csv_parser import TTKCsvReader, WeaponCsvReader


def main():
    # TODO: Measure TTK in terms of duration of your active firing (i.e. not counting short pauses).
    #  Active firing means counting one round period per round fired.
    #  i.e. You can multiply number of rounds fired with round period and add reload time if you're
    #  in the open when reloading.

    self_path = os.path.dirname(__file__)
    apex_stats_filename = os.path.join(self_path, 'Apex Legends Stats.csv')
    ttks_filename = os.path.join(self_path, 'Historic TTKs.csv')

    with open(apex_stats_filename, encoding='utf-8-sig') as fp:
        dr = WeaponCsvReader(fp)
        weapons: tuple[WeaponArchetype] = tuple(dr)

    with open(ttks_filename, encoding='utf-8-sig') as fp:
        dr = TTKCsvReader(fp)
        ttk_entries = tuple(dr)

    show_plots = True

    ttks = np.array(list(map(float, ttk_entries)))
    ttks.sort()
    ts_lin = np.linspace(ttks.min(), ttks.max(), num=1000)

    ts = ttks
    power = 0

    # ts = ts_lin
    # # power should be non-positive, probably in the range [0, -1]. Lower values means you're
    # # assuming TTK is going to be lower. Zero probably makes sense if your TTK values are good.
    # power = -1

    wingman: ConcreteWeapon = next(
        base_weapon
        for weapon_archetype in weapons
        for base_weapon in weapon_archetype.get_base_weapons(reload=False)
        if 'wingman' in base_weapon.get_name().lower())
    base_weapons: list[WeaponBase] = [
        base_weapon.combine(wingman)
        for weapon_archetype in weapons
        for base_weapon in weapon_archetype.get_base_weapons(reload=False)]

    for weapon_archetype in weapons:
        base_weapons.extend(weapon_archetype.get_base_weapons(reload=True))

    damage_table: np.ndarray[np.floating] = np.empty((len(base_weapons), len(ts)))
    for idx, base_weapon in enumerate(base_weapons):
        base_weapon: ConcreteWeapon = base_weapon

        # This could be made faster by using vectorized operations.
        damages_cum = np.array([base_weapon.get_cumulative_damage(t) for t in ts])
        damage_table[idx] = damages_cum

    # This is the mean dps up till time t for t in ttks.
    mean_dps_till_time_t = damage_table * (1 / ts)

    # This one actually weights earlier damage more highly.
    weighted_average_damage = np.average(mean_dps_till_time_t,
                                         axis=1,
                                         weights=(ts ** power))
    sorti = weighted_average_damage.argsort()[::-1]

    sorted_weapons = [base_weapons[idx] for idx in sorti]

    min_damage = weighted_average_damage.max() * .75
    n = np.count_nonzero(weighted_average_damage >= min_damage)
    sorted_weapons_str = '\n'.join(
        f'  {weighted_avg_damage:4.2f}: {weapon}'
        for idx, (weapon, weighted_avg_damage) in
        enumerate(zip(sorted_weapons[:n], weighted_average_damage[sorti[:n]]), start=1))
    print(f'Sorted Weapons:\n{sorted_weapons_str}')

    archetype: dict[WeaponArchetype, tuple[float, WeaponBase]] = {}
    for weapon, weighted_avg_damage in zip(sorted_weapons, weighted_average_damage[sorti]):
        if weapon.get_archetype() not in archetype:
            archetype[weapon.get_archetype()] = (weighted_avg_damage, weapon)

    archetypes = [
        f'  {weapon}: {dmg:4.2f}'
        for dmg, weapon in
        sorted(archetype.values(), key=lambda itm: itm[0], reverse=True)
        if dmg >= min_damage]
    archetypes_str = '\n'.join(archetypes)
    print(f'Sorted Weapon Archetypes:\n{archetypes_str}')

    if show_plots:
        for idx, base_weapon in zip(sorti[:n], sorted_weapons[:n]):
            damages = np.array([base_weapon.get_cumulative_damage(t) for t in ts_lin])
            damages *= 1 / ts_lin
            plt.plot(ts_lin, damages, label=base_weapon.get_name())

        plt.ylim((0, None))
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
