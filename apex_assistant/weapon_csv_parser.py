import abc
import csv
import logging
import math
from enum import StrEnum
from types import MappingProxyType
from typing import (Any,
                    Callable,
                    Generic,
                    IO,
                    Iterable, Iterator,
                    Mapping, Optional,
                    Tuple,
                    Type,
                    TypeAlias,
                    TypeVar, Union)

import numpy as np

from apex_assistant.checker import check_type
from apex_assistant.legend import Legend
from apex_assistant.speech.apex_terms import ARCHETYPES_TERM_TO_ARCHETYPE_SUFFIXES_DICT, Suffix
from apex_assistant.speech.term import RequiredTerm, Words
from apex_assistant.speech.term_translator import Translator
from apex_assistant.ttk_entry import Engagement
from apex_assistant.weapon import (MagazineCapacities,
                                   MagazineCapacity,
                                   RoundTiming,
                                   RoundTimingBurst,
                                   RoundTimingDevotion,
                                   RoundTimingHavoc,
                                   RoundTimingNemesis,
                                   RoundTimingRegular,
                                   RoundTimingType,
                                   RoundsPerMinute,
                                   RoundsPerMinutes,
                                   StockStat,
                                   StockStatValues,
                                   StockStats,
                                   StockStatsBase,
                                   WeaponArchetype)
from apex_assistant.weapon_class import WeaponClass

logger = logging.getLogger()
T = TypeVar('T')
_SUFFIX_T: TypeAlias = Union[Suffix, Tuple[Suffix, ...]]


class CsvReader(Generic[T]):
    CSV_DICT_TYPE = dict[str, str]

    def __init__(self, fp: IO):
        self._dict_reader: Iterable[CsvReader.CSV_DICT_TYPE] = csv.DictReader(fp)

    def __iter__(self) -> Iterator[T]:
        for row in self._dict_reader:
            row = CsvRow(row)
            try:
                item = self._parse_item(row)
            except (ValueError, KeyError) as ex:
                raise ex.__class__(f'Error parsing row: {row}: {ex}') from ex
            if item is not None:
                yield item

    @abc.abstractmethod
    def _parse_item(self, items: 'CsvRow') -> Optional[T]:
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
                error_message = key
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
                       enum_type: Type[T],
                       default_value: T | None = None,
                       error_message: str | None = None) -> Optional[T]:
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
                                 default_value=str(default_value).upper(),
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

    def __str__(self):
        return ','.join(map(str, self.row_dict.values()))


class EngagementCsvReader(CsvReader[Engagement]):
    KEY_START_FRAME = 'Start Frame'
    KEY_STOP_FRAME = 'Stop Frame'
    KEY_PAUSE_FRAMES = 'Pause Frames'
    KEY_FRAMES_PER_SECOND = 'FPS'
    KEY_ENEMY_HEIGHT = 'Enemy Height (pixels)'
    KEY_ENEMY_LEGEND = 'Enemy Legend'
    KEY_WINDOW_HEIGHT = 'Window Height (pixels)'
    KEY_FOV = 'FOV'

    # Source: https://www.youtube.com/shorts/KU0dROzSx1I
    _LEGEND_HEIGHTS_METERS: Mapping[Legend, float] = MappingProxyType({
        Legend.ASH: 1.85,
        Legend.BALLISTIC: 1.78,
        Legend.BANGALORE: 1.83,
        Legend.BLOODHOUND: 1.85,
        Legend.CAUSTIC: 1.91,
        Legend.CATALYST: 1.70,
        Legend.CONDUIT: 1.60,
        Legend.CRYPTO: 1.75,
        Legend.FUSE: 1.85,
        Legend.GIBRALTAR: 1.96,
        Legend.HORIZON: 1.83,
        Legend.LIFELINE: 1.65,
        Legend.LOBA: 1.73,
        Legend.MAD_MAGGIE: 1.83,
        Legend.MIRAGE: 1.80,
        Legend.NEWCASTLE: 1.98,
        Legend.OCTANE: 1.73,
        Legend.PATHFINDER: 1.88,
        Legend.RAMPART: 1.80,
        Legend.REVENANT: 2.03,
        Legend.SEER: 1.94,
        Legend.VALKYRIE: 1.68,
        Legend.VANTAGE: 1.73,
        Legend.WATTSON: 1.63,
        Legend.WRAITH: 1.63
    })
    _AVERAGE_LEGEND_HEIGHT = float(np.mean(list(_LEGEND_HEIGHTS_METERS.values())))

    @staticmethod
    def _get_legend_height_meters(legend: Legend) -> float:
        return EngagementCsvReader._LEGEND_HEIGHTS_METERS.get(
            legend,
            EngagementCsvReader._AVERAGE_LEGEND_HEIGHT)

    @staticmethod
    def _get_enemy_distance_meters(enemy_height_pixels: int,
                                   enemy_legend: Legend,
                                   window_height_pixels: int,
                                   fov_degrees: int) -> float:
        fov_radians = math.radians(fov_degrees)
        enemy_view_height_radians = fov_radians * (enemy_height_pixels / window_height_pixels)

        enemy_height_meters = EngagementCsvReader._get_legend_height_meters(enemy_legend)
        enemy_distance_meters = (enemy_height_meters / 2) / math.tan(enemy_view_height_radians / 2)
        return enemy_distance_meters

    def _parse_item(self, item: CsvRow) -> Engagement:
        start_frame = item.parse_int(self.KEY_START_FRAME)
        stop_frame = item.parse_int(self.KEY_STOP_FRAME)
        pause_frames = item.parse_int(self.KEY_PAUSE_FRAMES, default_value=0)
        frames_to_kill = stop_frame - start_frame - pause_frames

        fps = item.parse_float(self.KEY_FRAMES_PER_SECOND)
        ttff_seconds = frames_to_kill / fps

        enemy_height_pixels = item.parse_int(self.KEY_ENEMY_HEIGHT)
        enemy_legend = item.parse_str_enum(self.KEY_ENEMY_LEGEND, Legend)
        fov = item.parse_int(self.KEY_FOV)
        window_height_pixels = item.parse_int(self.KEY_WINDOW_HEIGHT)
        enemy_distance_meters = self._get_enemy_distance_meters(
            enemy_height_pixels=enemy_height_pixels,
            enemy_legend=enemy_legend,
            window_height_pixels=window_height_pixels,
            fov_degrees=fov)

        return Engagement(ttff_seconds=ttff_seconds,
                          enemy_distance_meters=enemy_distance_meters)


class WeaponCsvReader(CsvReader[WeaponArchetype]):
    KEY_WEAPON_ARCHETYPE = 'Weapon'
    KEY_WEAPON_CLASS = 'Weapon Class'
    KEY_LEGEND = 'Legend'
    KEY_80_PERCENT_ACCURACY_RANGE = '80% accuracy range'
    KEY_10_METER_ACCURACY = '10-meter accuracy'
    KEY_20_METER_ACCURACY = '20-meter accuracy'
    KEY_40_METER_ACCURACY = '40-meter accuracy'
    KEY_80_METER_ACCURACY = '80-meter accuracy'
    KEY_CARE_PACKAGE = 'Care Package'
    KEY_STOCKS_INCOMPATIBLE = 'Stocks Incompatible Override'
    KEY_DAMAGE_BODY = 'Damage (body)'
    KEY_RPM_BASE = 'RPM (base)'
    KEY_RPM_SHOTGUN_BOLT_LEVEL_1 = 'RPM (shotgun bolt level 1)'
    KEY_RPM_SHOTGUN_BOLT_LEVEL_2 = 'RPM (shotgun bolt level 2)'
    KEY_RPM_SHOTGUN_BOLT_LEVEL_3 = 'RPM (shotgun bolt level 3)'
    KEY_MAGAZINE_BASE = 'Magazine (base)'
    KEY_MAGAZINE_LEVEL_1 = 'Magazine (level 1)'
    KEY_MAGAZINE_LEVEL_2 = 'Magazine (level 2)'
    KEY_MAGAZINE_LEVEL_3 = 'Magazine (level 3)'
    KEY_ROUND_TIMING_TYPE = 'Round Timing'
    KEY_RPM_INITIAL = 'RPM initial'
    KEY_ROUNDS_PER_BURST = 'Rounds Per Burst'
    KEY_BURST_FIRE_DELAY = 'Burst Fire Delay'
    KEY_BURST_FIRE_DELAY_INITIAL = 'Burst Fire Delay Initial'
    KEY_BURST_CHARGE_FRACTION = 'Burst Charge Fraction'
    KEY_SPINUP_TIME = 'Spinup Time'
    KEY_HOLSTER_TIME = 'Holster Time'
    KEY_DEPLOY_TIME = 'Deploy Time'
    KEY_READY_TO_FIRE_TIME = 'Ready to Fire Time'
    KEY_ACTIVE = 'Active'
    KEY_TACTICAL_RELOAD_TIME_BASE = 'Tactical Reload Time (base)'
    KEY_TACTICAL_RELOAD_TIME_LEVEL_1 = 'Tactical Reload Time (level 1)'
    KEY_TACTICAL_RELOAD_TIME_LEVEL_2 = 'Tactical Reload Time (level 2)'
    KEY_TACTICAL_RELOAD_TIME_LEVEL_3 = 'Tactical Reload Time (level 3)'
    KEY_FULL_RELOAD_TIME_BASE = 'Full Reload Time (base)'
    KEY_FULL_RELOAD_TIME_LEVEL_1 = 'Full Reload Time (level 1)'
    KEY_FULL_RELOAD_TIME_LEVEL_2 = 'Full Reload Time (level 2)'
    KEY_FULL_RELOAD_TIME_LEVEL_3 = 'Full Reload Time (level 3)'
    KEY_HEAT_BASED = 'Heat Based'

    _ARCHETYPES_TRANSLATOR = Translator[_SUFFIX_T](ARCHETYPES_TERM_TO_ARCHETYPE_SUFFIXES_DICT)

    def __init__(self, fp: IO):
        super().__init__(fp)
        self._taken_terms: dict[Tuple[RequiredTerm, _SUFFIX_T], str] = {}

    @staticmethod
    def get_term_and_suffix(name: Words) -> Tuple[RequiredTerm, Optional[Suffix]]:
        check_type(Words, name=name)

        result: Optional[Tuple[RequiredTerm, Optional[Suffix]]] = None
        for translated_term in WeaponCsvReader._ARCHETYPES_TRANSLATOR.translate_terms(name):
            suffixes: _SUFFIX_T = translated_term.get_value()
            if suffixes is None:
                suffixes: _SUFFIX_T = tuple()
            elif not isinstance(suffixes, tuple):
                suffixes: _SUFFIX_T = (suffixes,)

            for suffix in suffixes:
                if not WeaponArchetype.find_suffix(suffix, translated_term.get_following_words()):
                    continue

                if result is not None:
                    logger.debug(f'More than one term for weapon archetype {name} found. We\'ll '
                                 'assume that the first one is right.')
                    return result

                result = translated_term.get_term(), suffix

            if result is None:
                result = translated_term.get_term(), None

        if result is None:
            raise RuntimeError(f'No term found for weapon archetype: {name}. Speech-to-text will '
                               'not work for it.')

        return result

    def _parse_weapon_archetype_term(self, item: CsvRow) -> tuple[str, RequiredTerm, Suffix]:
        # Figure out what term matches the weapon name.
        read_name = item.parse_str(self.KEY_WEAPON_ARCHETYPE)
        name_words = Words(read_name)
        tup = self.get_term_and_suffix(name_words)
        if tup in self._taken_terms:
            raise RuntimeError(f'Term {tup} refers to another weapon: {self._taken_terms[tup]}. '
                               f'Can\'t use it for {read_name}.')
        self._taken_terms[tup] = read_name

        term, suffix = tup
        return read_name, term, suffix

    def _parse_rpm(self, csv_row: CsvRow) -> RoundsPerMinute:
        rpm_base = csv_row.parse_float(self.KEY_RPM_BASE)
        error_message = 'Must specify either all shotgun bolt levels or just the base rpm.'
        if csv_row.has_value(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_1):
            rpm_shotgun_bolt_level_1 = csv_row.parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_1,
                                                           error_message=error_message)
            rpm_shotgun_bolt_level_2 = csv_row.parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_2,
                                                           error_message=error_message)
            rpm_shotgun_bolt_level_3 = csv_row.parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_3,
                                                           error_message=error_message)
            rpm = RoundsPerMinutes(base_rounds_per_minute=rpm_base,
                                   level_1_rounds_per_minute=rpm_shotgun_bolt_level_1,
                                   level_2_rounds_per_minute=rpm_shotgun_bolt_level_2,
                                   level_3_rounds_per_minute=rpm_shotgun_bolt_level_3)
        else:
            rpm = RoundsPerMinute(rpm_base)

        return rpm

    def _parse_mag(self, csv_row: CsvRow) -> MagazineCapacity:
        magazine_base = csv_row.parse_int(self.KEY_MAGAZINE_BASE)
        error_message = 'Must specify either all magazine levels or just the base magazine size.'
        if csv_row.has_value(self.KEY_MAGAZINE_LEVEL_1):
            magazine_level_1 = csv_row.parse_int(self.KEY_MAGAZINE_LEVEL_1,
                                                 error_message=error_message)
            magazine_level_2 = csv_row.parse_int(self.KEY_MAGAZINE_LEVEL_2,
                                                 error_message=error_message)
            magazine_level_3 = csv_row.parse_int(self.KEY_MAGAZINE_LEVEL_3,
                                                 error_message=error_message)
            mag = MagazineCapacities(base_capacity=magazine_base,
                                     level_1_capacity=magazine_level_1,
                                     level_2_capacity=magazine_level_2,
                                     level_3_capacity=magazine_level_3)
        else:
            mag = MagazineCapacity(magazine_base)
            csv_row.ensure_empty(error_message,
                                 self.KEY_MAGAZINE_LEVEL_1,
                                 self.KEY_MAGAZINE_LEVEL_2,
                                 self.KEY_MAGAZINE_LEVEL_3)

        return mag

    @staticmethod
    def _parse_reload_time(csv_row: CsvRow, key_base_reload_time) -> Optional[float]:
        if not csv_row.has_value(key_base_reload_time):
            return None
        return csv_row.parse_float(key_base_reload_time)

    @staticmethod
    def _parse_reload_times(csv_row: CsvRow,
                            key_base_reload_time,
                            key_level_1_reload_time,
                            key_level_2_reload_time,
                            key_level_3_reload_time) -> Tuple[float, ...]:
        return (csv_row.parse_float(key_base_reload_time),
                csv_row.parse_float(key_level_1_reload_time),
                csv_row.parse_float(key_level_2_reload_time),
                csv_row.parse_float(key_level_3_reload_time))

    def _parse_stock_dependant_stats(self, row: CsvRow, stock_compatible: bool) -> StockStatsBase:
        holster_time_secs_base = row.parse_float(self.KEY_HOLSTER_TIME)
        ready_to_fire_time_secs_base = row.parse_float(self.KEY_READY_TO_FIRE_TIME)

        if stock_compatible:
            tactical_reloads = self._parse_reload_times(
                row,
                key_base_reload_time=self.KEY_TACTICAL_RELOAD_TIME_BASE,
                key_level_1_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_1,
                key_level_2_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_2,
                key_level_3_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_3)
            full_reloads = self._parse_reload_times(
                row,
                key_base_reload_time=self.KEY_FULL_RELOAD_TIME_BASE,
                key_level_1_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_1,
                key_level_2_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_2,
                key_level_3_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_3)

            # Multipliers for holster/deploy/RTF times of level 0/1/2/3 stocks.
            stock_multipliers = (1, 0.85, 0.8, 0.75)

            stock_stats = StockStats(*(
                StockStatValues(
                    tactical_reload_time_secs=tac_reload,
                    full_reload_time_secs=full_reload,
                    holster_time_secs=holster_time_secs_base * multiplier,
                    ready_to_fire_time_secs=ready_to_fire_time_secs_base * multiplier)
                for tac_reload, full_reload, multiplier in zip(tactical_reloads,
                                                               full_reloads,
                                                               stock_multipliers)))
        else:
            tactical_reload = self._parse_reload_time(
                row,
                key_base_reload_time=self.KEY_TACTICAL_RELOAD_TIME_BASE)
            full_reload = self._parse_reload_time(
                row,
                key_base_reload_time=self.KEY_FULL_RELOAD_TIME_BASE)
            row.ensure_empty(
                f'Cannot specify non-base tactical or full reload times for {row}',
                self.KEY_TACTICAL_RELOAD_TIME_LEVEL_1,
                self.KEY_TACTICAL_RELOAD_TIME_LEVEL_2,
                self.KEY_TACTICAL_RELOAD_TIME_LEVEL_3,
                self.KEY_FULL_RELOAD_TIME_LEVEL_1,
                self.KEY_FULL_RELOAD_TIME_LEVEL_2,
                self.KEY_FULL_RELOAD_TIME_LEVEL_3)

            stock_stats = StockStat(StockStatValues(
                tactical_reload_time_secs=tactical_reload,
                full_reload_time_secs=full_reload,
                holster_time_secs=holster_time_secs_base,
                ready_to_fire_time_secs=ready_to_fire_time_secs_base))

        return stock_stats

    def _parse_spinup(self, csv_row: CsvRow) -> RoundTiming:
        spinup_type = csv_row.parse_str_enum(self.KEY_ROUND_TIMING_TYPE,
                                             RoundTimingType,
                                             RoundTimingType.NONE)

        disallowed_keys: set[str] = {self.KEY_RPM_INITIAL,
                                     self.KEY_SPINUP_TIME,
                                     self.KEY_ROUNDS_PER_BURST,
                                     self.KEY_BURST_FIRE_DELAY,
                                     self.KEY_BURST_FIRE_DELAY_INITIAL,
                                     self.KEY_BURST_CHARGE_FRACTION}
        if spinup_type is RoundTimingType.NONE:
            spinup = RoundTimingRegular.get_instance()

        elif spinup_type is RoundTimingType.DEVOTION:
            rpm_initial = csv_row.parse_float(self.KEY_RPM_INITIAL)
            spinup_time = csv_row.parse_float(self.KEY_SPINUP_TIME)
            spinup = RoundTimingDevotion(rounds_per_minute_initial=rpm_initial,
                                         spinup_time_seconds=spinup_time)
            disallowed_keys -= {self.KEY_RPM_INITIAL, self.KEY_SPINUP_TIME}

        elif spinup_type is RoundTimingType.DELAY:
            spinup_time = csv_row.parse_float(self.KEY_SPINUP_TIME)
            spinup = RoundTimingHavoc(spinup_time)
            disallowed_keys.remove(self.KEY_SPINUP_TIME)

        elif spinup_type is RoundTimingType.BURST:
            rounds_per_burst = csv_row.parse_int(self.KEY_ROUNDS_PER_BURST)
            burst_fire_delay = csv_row.parse_float(self.KEY_BURST_FIRE_DELAY)
            spinup = RoundTimingBurst(rounds_per_burst=rounds_per_burst,
                                      burst_fire_delay=burst_fire_delay)
            disallowed_keys -= {self.KEY_ROUNDS_PER_BURST, self.KEY_BURST_FIRE_DELAY}

        elif spinup_type is RoundTimingType.NEMESIS:
            rounds_per_burst = csv_row.parse_int(self.KEY_ROUNDS_PER_BURST)
            burst_fire_delay_initial = csv_row.parse_float(self.KEY_BURST_FIRE_DELAY_INITIAL)
            burst_fire_delay_final = csv_row.parse_float(self.KEY_BURST_FIRE_DELAY)
            burst_charge_fraction = csv_row.parse_float(self.KEY_BURST_CHARGE_FRACTION)
            spinup = RoundTimingNemesis(rounds_per_burst=rounds_per_burst,
                                        burst_fire_delay_initial=burst_fire_delay_initial,
                                        burst_fire_delay_final=burst_fire_delay_final,
                                        burst_charge_fraction=burst_charge_fraction)
            disallowed_keys -= {self.KEY_ROUNDS_PER_BURST,
                                self.KEY_BURST_FIRE_DELAY,
                                self.KEY_BURST_FIRE_DELAY_INITIAL,
                                self.KEY_BURST_CHARGE_FRACTION}

        else:
            raise ValueError(f'{self.KEY_ROUND_TIMING_TYPE} of {spinup_type} is unsupported.')

        csv_row.ensure_empty(
            lambda key: f'{key} must be empty for spinup type of {spinup_type}',
            *disallowed_keys)

        return spinup

    def _parse_item(self, row: CsvRow) -> Optional[WeaponArchetype]:
        active = row.parse_bool(self.KEY_ACTIVE)
        if not active:
            name = row.parse_str(self.KEY_WEAPON_ARCHETYPE)
            logger.info(f'Weapon "{name}" is not active. Skipping.')
            return None

        name, term, suffix = self._parse_weapon_archetype_term(row)
        weapon_class: WeaponClass = row.parse_str_enum(key=self.KEY_WEAPON_CLASS,
                                                       enum_type=WeaponClass)
        legend = (row.parse_str_enum(self.KEY_LEGEND, Legend)
                  if row.has_value(self.KEY_LEGEND)
                  else None)
        eighty_percent_accuracy_range = row.parse_int(self.KEY_80_PERCENT_ACCURACY_RANGE)
        ten_meter_accuracy = row.parse_float(self.KEY_10_METER_ACCURACY)
        twenty_meter_accuracy = row.parse_float(self.KEY_20_METER_ACCURACY)
        forty_meter_accuracy = row.parse_float(self.KEY_40_METER_ACCURACY)
        eighty_meter_accuracy = row.parse_float(self.KEY_80_METER_ACCURACY)
        dist_to_accuracy_mapping: Mapping[int, float] = {
            10: ten_meter_accuracy,
            20: twenty_meter_accuracy,
            40: forty_meter_accuracy,
            80: eighty_meter_accuracy
        }
        damage_body = row.parse_float(self.KEY_DAMAGE_BODY)

        rpm = self._parse_rpm(row)
        mag = self._parse_mag(row)

        care_package = row.parse_bool(key=self.KEY_CARE_PACKAGE, default_value=False)
        stock_incompatible = row.parse_bool(
            key=self.KEY_STOCKS_INCOMPATIBLE,
            default_value=care_package or weapon_class is WeaponClass.PISTOL)
        stock_compatible = not stock_incompatible

        stock_dependant_stats = self._parse_stock_dependant_stats(row,
                                                                  stock_compatible=stock_compatible)
        spinup = self._parse_spinup(row)
        heat_based = row.parse_bool(self.KEY_HEAT_BASED, default_value=False)

        return WeaponArchetype(name=name,
                               base_term=term,
                               suffix=suffix,
                               weapon_class=weapon_class,
                               eighty_percent_accuracy_range=eighty_percent_accuracy_range,
                               dist_to_accuracy_mapping=dist_to_accuracy_mapping,
                               damage_body=damage_body,
                               rounds_per_minute=rpm,
                               magazine_capacity=mag,
                               stock_dependant_stats=stock_dependant_stats,
                               spinup=spinup,
                               heat_based=heat_based,
                               associated_legend=legend,
                               care_package=care_package)
