import abc
import csv
import logging
from enum import StrEnum
from typing import (Any,
                    Callable,
                    Generic,
                    IO,
                    Iterator,
                    Optional,
                    Tuple,
                    Type,
                    TypeAlias,
                    TypeVar,
                    Union)

from apex_assistant.checker import check_type
from apex_assistant.speech.apex_terms import ARCHETYPES_TERM_TO_ARCHETYPE_SUFFIX_DICT
from apex_assistant.speech.term import OptTerm, RequiredTerm, TermBase, Words
from apex_assistant.speech.term_translator import (Translator)
from apex_assistant.ttk_datum import TTKDatum
from apex_assistant.weapon import (MagazineCapacity,
                                   ReloadTime,
                                   RoundsPerMinute,
                                   SingleMagazineCapacity,
                                   SingleReloadTime,
                                   SingleRoundsPerMinute,
                                   Spinup,
                                   SpinupDevotion,
                                   SpinupHavoc,
                                   SpinupNone,
                                   SpinupType,
                                   WeaponArchetype, WeaponArchetypes)
from apex_assistant.weapon_comparer import WeaponComparer

logger = logging.getLogger()
T = TypeVar('T')
_SUFFIX_T: TypeAlias = Optional[TermBase]


class CsvReader(Generic[T]):
    CSV_DICT_TYPE = dict[str, str]

    def __init__(self, fp: IO):
        self._dict_reader = csv.DictReader(fp)

    def __iter__(self) -> Iterator[T]:
        for item in self._parse_items(iter(map(CsvRow, self._dict_reader))):
            if item is not None:
                yield item

    @abc.abstractmethod
    def _parse_items(self, items: Iterator['CsvRow']) -> Iterator[T]:
        raise NotImplementedError()


class CsvRow:
    VALUE_EMPTY = ''

    def __init__(self, row_dict: CsvReader.CSV_DICT_TYPE):
        self.row_dict = row_dict

    def _parse(self,
               key: str,
               default_value: Any | None,
               clazz: Any,
               error_message: str | None):
        val_str = self.row_dict[key]
        if val_str != self.VALUE_EMPTY:
            try:
                return clazz(val_str)
            except ValueError as ex:
                raise ValueError(f'Value for {key} must be of type {clazz.__name__}') from ex
        elif default_value is None:
            if error_message is None:
                row_str = ' | '.join(map(str, self.row_dict.values()))
                error_message = f'Missing value for "{key}" in row: {row_str}.'
            raise KeyError(error_message)
        else:
            return default_value

    def has_value(self, key: str) -> bool:
        return self.row_dict[key] != self.VALUE_EMPTY

    def parse_str(self,
                  key: str,
                  default_value: str | None = None,
                  error_message: str | None = None) -> str:
        return self._parse(key=key,
                           default_value=default_value,
                           clazz=str,
                           error_message=error_message)

    def parse_str_enum(self,
                       key: str,
                       enum_type: Type[StrEnum],
                       default_value: StrEnum | None = None,
                       error_message: str | None = None):
        assert issubclass(enum_type, StrEnum)
        enum_val = self._parse(key=key,
                               default_value=default_value,
                               clazz=enum_type,
                               error_message=error_message)
        return enum_val

    def parse_float(self,
                    key: str,
                    default_value: float | None = None,
                    error_message: str | None = None) -> float:
        return self._parse(key=key,
                           default_value=default_value,
                           clazz=float,
                           error_message=error_message)

    def parse_int(self,
                  key: str,
                  default_value: int | None = None,
                  error_message: str | None = None):
        return self._parse(key=key,
                           default_value=default_value,
                           clazz=int,
                           error_message=error_message)

    def parse_bool(self,
                   key: str,
                   default_value: bool | None = None,
                   error_message: str | None = None):
        str_val = self.parse_str(key=key,
                                 default_value=str(default_value),
                                 error_message=error_message)
        if str_val == 'TRUE':
            return True
        elif str_val == 'FALSE':
            return False
        else:
            raise ValueError(f'Value for {key} must be either TRUE or FALSE')

    def ensure_empty(self,
                     error_message: str | Callable[[str], str],
                     *keys: str):
        for key in keys:
            is_empty = not self.has_value(key=key)
            if not is_empty:
                if not isinstance(error_message, str):
                    error_message = error_message(key)
                raise ValueError(error_message)


class TTKCsvReader(CsvReader[TTKDatum]):
    KEY_CLIP = 'Clip of classic blunder'
    KEY_FRAMES_TO_KILL = 'Frames to Kill'
    KEY_FPS = 'FPS'

    def _parse_items(self, items: Iterator['CsvRow']) -> Iterator[TTKDatum]:
        for item in items:
            try:
                clip_name = item.parse_str(self.KEY_CLIP)
            except KeyError:
                clip_name = None
            frames_to_kill = item.parse_int(self.KEY_FRAMES_TO_KILL)
            fps = item.parse_float(self.KEY_FPS)
            ttk_seconds = frames_to_kill / fps
            yield TTKDatum(clip_name, ttk_seconds)


class WeaponCsvReader(CsvReader[WeaponArchetypes]):
    KEY_WEAPON_ARCHETYPE = "Weapon"
    KEY_WEAPON_CLASS = "Weapon Class"
    KEY_DAMAGE_BODY = "Damage (body)"
    KEY_RPM_BASE = "RPM (base)"
    KEY_RPM_SHOTGUN_BOLT_LEVEL_1 = "RPM (shotgun bolt level 1)"
    KEY_RPM_SHOTGUN_BOLT_LEVEL_2 = "RPM (shotgun bolt level 2)"
    KEY_RPM_SHOTGUN_BOLT_LEVEL_3 = "RPM (shotgun bolt level 3)"
    KEY_MAGAZINE_BASE = "Magazine (base)"
    KEY_MAGAZINE_LEVEL_1 = "Magazine (level 1)"
    KEY_MAGAZINE_LEVEL_2 = "Magazine (level 2)"
    KEY_MAGAZINE_LEVEL_3 = "Magazine (level 3)"
    KEY_SPINUP_TYPE = "Spinup Type"
    KEY_RPM_INITIAL = "RPM initial"
    KEY_SPINUP_TIME = "Spinup Time"
    KEY_DEPLOY_TIME = "Deploy Time"
    KEY_ACTIVE = "Active"
    KEY_TACTICAL_RELOAD_TIME_BASE = 'Tactical Reload Time (base)'
    KEY_TACTICAL_RELOAD_TIME_LEVEL_1 = 'Tactical Reload Time (level 1)'
    KEY_TACTICAL_RELOAD_TIME_LEVEL_2 = 'Tactical Reload Time (level 2)'
    KEY_TACTICAL_RELOAD_TIME_LEVEL_3 = 'Tactical Reload Time (level 3)'
    KEY_FULL_RELOAD_TIME_BASE = 'Full Reload Time (base)'
    KEY_FULL_RELOAD_TIME_LEVEL_1 = 'Full Reload Time (level 1)'
    KEY_FULL_RELOAD_TIME_LEVEL_2 = 'Full Reload Time (level 2)'
    KEY_FULL_RELOAD_TIME_LEVEL_3 = 'Full Reload Time (level 3)'

    _ARCHETYPE_TRANSLATOR = Translator(ARCHETYPES_TERM_TO_ARCHETYPE_SUFFIX_DICT)

    def __init__(self, fp: IO, weapon_comparer: WeaponComparer):
        check_type(WeaponComparer, weapon_comparer=weapon_comparer)
        super().__init__(fp)
        self._taken_terms: set[Tuple[RequiredTerm, _SUFFIX_T]] = set()
        self._comparer = weapon_comparer

    @staticmethod
    def get_term_and_suffix(name: Words) -> Tuple[RequiredTerm, _SUFFIX_T]:
        check_type(Words, name=name)

        result: Optional[Tuple[RequiredTerm, _SUFFIX_T]] = None
        warning = (f'More than one term for weapon archetype {name} found. We\'ll assume that the '
                   'first one is right.')
        for translated_term in WeaponCsvReader._ARCHETYPE_TRANSLATOR.translate_terms(name):
            has_none: bool = False
            for suffix in translated_term.get_value():
                if suffix is None:
                    has_none = True
                elif suffix.has_variation(translated_term.get_following_words()):
                    if result is not None:
                        logging.warning(warning)
                        return result

                    result = translated_term.get_term(), suffix

            if has_none and result is None:
                result = translated_term.get_term(), None

        if result is None:
            raise RuntimeError(f'No term found for weapon archetype: {name}. Speech-to-text '
                               'will not work for it.')

        return result

    def _parse_weapon_archetype_term(self, item: CsvRow, key: str) -> \
            tuple[str, RequiredTerm, Optional[Union[RequiredTerm, OptTerm]]]:
        # Figure out what term matches the weapon name.
        read_name = item.parse_str(key)
        name_words = Words(read_name)
        tup = self.get_term_and_suffix(name_words)
        if tup in self._taken_terms:
            raise RuntimeError(f'Term {tup} refers to another weapon. Can\'t use it.')
        self._taken_terms.add(tup)

        term, suffix = tup
        return read_name, term, suffix

    def _parse_rpm(self, csv_row: CsvRow) -> SingleRoundsPerMinute:
        rpm_base = csv_row.parse_float(self.KEY_RPM_BASE)
        error_message = 'Must specify either all shotgun bolt levels or just the base rpm.'
        if csv_row.has_value(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_1):
            rpm_shotgun_bolt_level_1 = csv_row.parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_1,
                                                           error_message=error_message)
            rpm_shotgun_bolt_level_2 = csv_row.parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_2,
                                                           error_message=error_message)
            rpm_shotgun_bolt_level_3 = csv_row.parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_3,
                                                           error_message=error_message)
            rpm = RoundsPerMinute(base_rounds_per_minute=rpm_base,
                                  level_1_rounds_per_minute=rpm_shotgun_bolt_level_1,
                                  level_2_rounds_per_minute=rpm_shotgun_bolt_level_2,
                                  level_3_rounds_per_minute=rpm_shotgun_bolt_level_3)
        else:
            rpm = SingleRoundsPerMinute(rpm_base)

        return rpm

    def _parse_mag(self, csv_row: CsvRow) -> SingleMagazineCapacity:
        magazine_base = csv_row.parse_int(self.KEY_MAGAZINE_BASE)
        error_message = 'Must specify either all magazine levels or just the base magazine size.'
        if csv_row.has_value(self.KEY_MAGAZINE_LEVEL_1):
            magazine_level_1 = csv_row.parse_int(self.KEY_MAGAZINE_LEVEL_1,
                                                 error_message=error_message)
            magazine_level_2 = csv_row.parse_int(self.KEY_MAGAZINE_LEVEL_2,
                                                 error_message=error_message)
            magazine_level_3 = csv_row.parse_int(self.KEY_MAGAZINE_LEVEL_3,
                                                 error_message=error_message)
            mag = MagazineCapacity(base_capacity=magazine_base,
                                   level_1_capacity=magazine_level_1,
                                   level_2_capacity=magazine_level_2,
                                   level_3_capacity=magazine_level_3)
        else:
            mag = SingleMagazineCapacity(magazine_base)
            csv_row.ensure_empty(error_message,
                                 self.KEY_MAGAZINE_LEVEL_1,
                                 self.KEY_MAGAZINE_LEVEL_2,
                                 self.KEY_MAGAZINE_LEVEL_3)

        return mag

    def _parse_reload_time(self,
                           csv_row: CsvRow,
                           key_base_reload_time,
                           key_level_1_reload_time,
                           key_level_2_reload_time,
                           key_level_3_reload_time):
        error_message = ('Must specify no reload times; base tactical and full reload times only; '
                         'or all tactical and full reload times.')
        if not csv_row.has_value(key_base_reload_time):
            reload = SingleReloadTime(None)
        elif not csv_row.has_value(key_level_1_reload_time):
            reload = SingleReloadTime(csv_row.parse_float(key_base_reload_time))
            csv_row.ensure_empty(error_message, key_level_2_reload_time, key_level_3_reload_time)
        else:
            reload = ReloadTime(
                base_reload_time_secs=csv_row.parse_float(self.KEY_TACTICAL_RELOAD_TIME_BASE,
                                                          error_message=error_message),
                level_1_reload_time_secs=csv_row.parse_float(key_level_1_reload_time,
                                                             error_message=error_message),
                level_2_reload_time_secs=csv_row.parse_float(key_level_2_reload_time,
                                                             error_message=error_message),
                level_3_reload_time_secs=csv_row.parse_float(key_level_3_reload_time,
                                                             error_message=error_message))

        return reload

    def _parse_tactical_and_full_reload_time(self, row: CsvRow) -> \
            tuple[SingleReloadTime, SingleReloadTime]:
        tactical_reload = self._parse_reload_time(
            row,
            key_base_reload_time=self.KEY_TACTICAL_RELOAD_TIME_BASE,
            key_level_1_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_1,
            key_level_2_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_2,
            key_level_3_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_3)
        full_reload = self._parse_reload_time(
            row,
            key_base_reload_time=self.KEY_FULL_RELOAD_TIME_BASE,
            key_level_1_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_1,
            key_level_2_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_2,
            key_level_3_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_3)
        return tactical_reload, full_reload

    def _parse_spinup(self, csv_row: CsvRow) -> Spinup:
        spinup_type = csv_row.parse_str_enum(self.KEY_SPINUP_TYPE,
                                             SpinupType,
                                             SpinupType.NONE)
        if spinup_type is SpinupType.NONE:
            spinup = SpinupNone()
            csv_row.ensure_empty(
                lambda key: f'{key} must be empty for spinup type of {spinup_type}',
                self.KEY_RPM_INITIAL,
                self.KEY_SPINUP_TIME)
        elif spinup_type is SpinupType.DEVOTION:
            rpm_initial = csv_row.parse_float(self.KEY_RPM_INITIAL)
            spinup_time = csv_row.parse_float(self.KEY_SPINUP_TIME)
            spinup = SpinupDevotion(rounds_per_minute_initial=rpm_initial,
                                    spinup_time_seconds=spinup_time)
        elif spinup_type is SpinupType.HAVOC:
            spinup_time = csv_row.parse_float(self.KEY_SPINUP_TIME)
            csv_row.ensure_empty(
                lambda key: f'{key} must be empty for spinup type of {spinup_type}',
                self.KEY_RPM_INITIAL)
            spinup = SpinupHavoc(spinup_time)
        else:
            raise ValueError(f'Spinup of {spinup_type} is unsupported.')

        return spinup

    def _parse_items(self, rows: Iterator[CsvRow]) -> Iterator[WeaponArchetypes]:
        weapon_archetypes_dict: dict[RequiredTerm, list[WeaponArchetype]] = {}

        for row in rows:
            active = row.parse_bool(self.KEY_ACTIVE)
            if not active:
                name = row.parse_str(self.KEY_WEAPON_ARCHETYPE)
                logger.info(f'Weapon "{name}" is not active. Skipping.')
                continue

            # Parse basic stats
            name, term, suffix = self._parse_weapon_archetype_term(row, self.KEY_WEAPON_ARCHETYPE)
            weapon_class = row.parse_str(self.KEY_WEAPON_CLASS, default_value='')
            damage_body = row.parse_float(self.KEY_DAMAGE_BODY)
            deploy_time_secs = row.parse_float(self.KEY_DEPLOY_TIME)

            rpm = self._parse_rpm(row)
            mag = self._parse_mag(row)
            tactical_reload_time, full_reload_time = self._parse_tactical_and_full_reload_time(row)
            spinup = self._parse_spinup(row)

            archetype = WeaponArchetype(name=name,
                                        base_term=term,
                                        suffix=suffix,
                                        weapon_class=weapon_class,
                                        damage_body=damage_body,
                                        deploy_time_secs=deploy_time_secs,
                                        rounds_per_minute=rpm,
                                        magazine_capacity=mag,
                                        tactical_reload_time=tactical_reload_time,
                                        full_reload_time=full_reload_time,
                                        spinup=spinup)

            if term not in weapon_archetypes_dict:
                archetypes = []
                weapon_archetypes_dict[term] = archetypes
            else:
                archetypes = weapon_archetypes_dict[term]
            archetypes.append(archetype)

        for base_term, archetypes in weapon_archetypes_dict.items():
            yield WeaponArchetypes(tuple(sorted(archetypes,
                                                key=self.get_expected_mean_dps,
                                                reverse=True)))

    def get_expected_mean_dps(self, archetype: WeaponArchetype):
        return self._comparer.get_expected_mean_dps(archetype.get_best_weapon())
