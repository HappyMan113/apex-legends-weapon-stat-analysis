from typing import AbstractSet, Dict, FrozenSet, Generator, Iterable, Tuple

from tqdm import tqdm

from apex_assistant.weapon import FullLoadout, Weapon

__CACHED_LOADOUTS: Dict[FrozenSet[Weapon], FullLoadout] = {}


def get_or_create_loadout(weapon_a: Weapon, weapon_b: Weapon) -> FullLoadout:
    hash_code = frozenset((weapon_a, weapon_b))
    if hash_code in __CACHED_LOADOUTS:
        return __CACHED_LOADOUTS[hash_code]

    loadout = FullLoadout(weapon_a, weapon_b)
    __CACHED_LOADOUTS[hash_code] = loadout
    return loadout


def get_loadouts(required_weapons: Iterable['Weapon']) -> Generator['FullLoadout', None, None]:
    base_generator = (get_or_create_loadout(weapon_a, weapon_b)
                      for weapon_a in required_weapons
                      for weapon_b in required_weapons)
    return base_generator


def filter_loadouts(loadouts: Iterable['FullLoadout'],
                    weapons_to_exclude: AbstractSet['Weapon']) -> \
        Generator['FullLoadout', None, None]:
    return (loadout
            for loadout in loadouts
            if loadout.get_weapons().isdisjoint(weapons_to_exclude))


def preload_loadouts(weapons: Tuple[Weapon, ...]):
    fully_kitted_loadouts = tuple(get_loadouts(weapons))
    with tqdm(total=len(fully_kitted_loadouts), ncols=80) as t:
        for loadout in fully_kitted_loadouts:
            loadout.lazy_load_internals()
            t.update()
