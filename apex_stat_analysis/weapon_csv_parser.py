import abc
import csv
import typing
from enum import StrEnum
from typing import Any, Generic, IO, Type, TypeVar

from apex_stat_analysis.ttk_datum import TTKDatum
from apex_stat_analysis.weapon import (MagazineCapacity,
                                       ReloadTime,
                                       RoundsPerMinute,
                                       SpinupDevotion,
                                       SpinupHavoc,
                                       SingleMagazineCapacity,
                                       SpinupNone,
                                       SingleReloadTime,
                                       SingleRoundsPerMinute,
                                       Spinup,
                                       SpinupType,
                                       WeaponArchetype)


T = TypeVar('T')

class CsvReader(Generic[T]):
    VALUE_EMPTY = ''
    WEAPON_DICT_TYPE = dict[str, str]

    def __init__(self, fp: IO):
        self._dict_reader = csv.DictReader(fp)
        self._cur_dict: CsvReader.WEAPON_DICT_TYPE | None = None

    def _parse(self,
               key: str,
               default_value: Any | None,
               clazz: Any,
               error_message: str | None):
        val_str = self._cur_dict[key]
        if val_str != self.VALUE_EMPTY:
            try:
                return clazz(val_str)
            except ValueError as ex:
                raise ValueError(f'Value for {key} must be of type {clazz.__name__}') from ex
        elif default_value is None:
            if error_message is None:
                error_message = f'Missing value for {key} in {self._cur_dict}.'
            raise KeyError(error_message)
        else:
            return default_value

    def _has_value(self, key):
        return self._cur_dict[key] != self.VALUE_EMPTY

    def _parse_str(self,
                   key: str,
                   default_value: str | None = None,
                   error_message: str | None = None) -> str:
        return self._parse(key,
                           default_value=default_value,
                           clazz=str,
                           error_message=error_message)

    def _parse_str_enum(self,
                        key: str,
                        enum_type: Type[StrEnum],
                        default_value: StrEnum | None = None,
                        error_message: str | None = None):
        assert issubclass(enum_type, StrEnum)
        enum_val = self._parse(key,
                               default_value=default_value,
                               clazz=enum_type,
                               error_message=error_message)
        return enum_val

    def _parse_float(self,
                     key: str,
                     default_value: float | None = None,
                     error_message: str | None = None) -> float:
        return self._parse(key,
                           default_value=default_value,
                           clazz=float,
                           error_message=error_message)

    def _parse_int(self,
                   key: str,
                   default_value: int | None = None,
                   error_message: str | None = None):
        return self._parse(key,
                           default_value=default_value,
                           clazz=int,
                           error_message=error_message)

    def _parse_bool(self,
                    key: str,
                    default_value: bool | None = None,
                    error_message: str | None = None):
        str_val = self._parse_str(key,
                                  default_value=str(default_value),
                                  error_message=error_message)
        if str_val == 'TRUE':
            return True
        elif str_val == 'FALSE':
            return False
        else:
            raise ValueError(f'Value for {key} must be either TRUE or FALSE')

    def _ensure_empty(self,
                      error_message: str | typing.Callable[[str], str],
                      *keys: str):
        for key in keys:
            is_empty = not self._has_value(key)
            if not is_empty:
                if not isinstance(error_message, str):
                    error_message = error_message(key)
                raise ValueError(error_message)

    def __iter__(self) -> typing.Iterator[T]:
        for item in self._dict_reader.__iter__():
            self._cur_dict = item
            yield self._parse_item()

    @abc.abstractmethod
    def _parse_item(self) -> T:
        raise NotImplementedError()


class TTKCsvReader(CsvReader[TTKDatum]):
    KEY_CLIP = 'Clip of classic blunder'
    KEY_FRAMES_TO_KILL = 'Frames to Kill'
    KEY_FPS = 'FPS'

    def _parse_item(self) -> TTKDatum:
        try:
            clip_name = self._parse_str(self.KEY_CLIP)
        except KeyError:
            clip_name = None
        frames_to_kill = self._parse_int(self.KEY_FRAMES_TO_KILL)
        fps = self._parse_float(self.KEY_FPS)
        ttk_seconds = frames_to_kill / fps
        return TTKDatum(clip_name, ttk_seconds)


class WeaponCsvReader(CsvReader[WeaponArchetype]):
    KEY_WEAPON = "Weapon"
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

    def __init__(self, fp: typing.IO):
        super().__init__(fp)

    def _parse_rpm(self) -> SingleRoundsPerMinute:
        rpm_base = self._parse_float(self.KEY_RPM_BASE)
        error_message = 'Must specify either all shotgun bolt levels or just the base rpm.'
        if self._has_value(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_1):
            rpm_shotgun_bolt_level_1 = self._parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_1,
                                                         error_message=error_message)
            rpm_shotgun_bolt_level_2 = self._parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_2,
                                                         error_message=error_message)
            rpm_shotgun_bolt_level_3 = self._parse_float(self.KEY_RPM_SHOTGUN_BOLT_LEVEL_3,
                                                         error_message=error_message)
            rpm = RoundsPerMinute(base_rounds_per_minute=rpm_base,
                                  level_1_rounds_per_minute=rpm_shotgun_bolt_level_1,
                                  level_2_rounds_per_minute=rpm_shotgun_bolt_level_2,
                                  level_3_rounds_per_minute=rpm_shotgun_bolt_level_3)
        else:
            rpm = SingleRoundsPerMinute(rpm_base)

        return rpm

    def _parse_mag(self) -> SingleMagazineCapacity:
        magazine_base = self._parse_int(self.KEY_MAGAZINE_BASE)
        error_message = 'Must specify either all magazine levels or just the base magazine size.'
        if self._has_value(self.KEY_MAGAZINE_LEVEL_1):
            magazine_level_1 = self._parse_int(self.KEY_MAGAZINE_LEVEL_1,
                                               error_message=error_message)
            magazine_level_2 = self._parse_int(self.KEY_MAGAZINE_LEVEL_2,
                                               error_message=error_message)
            magazine_level_3 = self._parse_int(self.KEY_MAGAZINE_LEVEL_3,
                                               error_message=error_message)
            mag = MagazineCapacity(base_capacity=magazine_base,
                                   level_1_capacity=magazine_level_1,
                                   level_2_capacity=magazine_level_2,
                                   level_3_capacity=magazine_level_3)
        else:
            mag = SingleMagazineCapacity(magazine_base)
            self._ensure_empty(error_message,
                               self.KEY_MAGAZINE_LEVEL_1,
                               self.KEY_MAGAZINE_LEVEL_2,
                               self.KEY_MAGAZINE_LEVEL_3)

        return mag

    def _parse_reload_time(self,
                           key_base_reload_time,
                           key_level_1_reload_time,
                           key_level_2_reload_time,
                           key_level_3_reload_time):
        error_message = ('Must specify no reload times; base tactical and full reload times only; '
                         'or all tactical and full reload times.')
        if not self._has_value(key_base_reload_time):
            reload = SingleReloadTime(None)
        elif not self._has_value(key_level_1_reload_time):
            reload = SingleReloadTime(self._parse_float(key_base_reload_time))
            self._ensure_empty(error_message, key_level_2_reload_time, key_level_3_reload_time)
        else:
            reload = ReloadTime(
                base_reload_time_secs=self._parse_float(self.KEY_TACTICAL_RELOAD_TIME_BASE,
                                                        error_message=error_message),
                level_1_reload_time_secs=self._parse_float(key_level_1_reload_time,
                                                           error_message=error_message),
                level_2_reload_time_secs=self._parse_float(key_level_2_reload_time,
                                                           error_message=error_message),
                level_3_reload_time_secs=self._parse_float(key_level_3_reload_time,
                                                           error_message=error_message))

        return reload

    def _parse_tactical_and_full_reload_time(self) -> tuple[SingleReloadTime, SingleReloadTime]:
        tactical_reload = self._parse_reload_time(
            key_base_reload_time=self.KEY_TACTICAL_RELOAD_TIME_BASE,
            key_level_1_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_1,
            key_level_2_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_2,
            key_level_3_reload_time=self.KEY_TACTICAL_RELOAD_TIME_LEVEL_3)
        full_reload = self._parse_reload_time(
            key_base_reload_time=self.KEY_FULL_RELOAD_TIME_BASE,
            key_level_1_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_1,
            key_level_2_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_2,
            key_level_3_reload_time=self.KEY_FULL_RELOAD_TIME_LEVEL_3)
        return tactical_reload, full_reload

    def _parse_spinup(self) -> Spinup:
        spinup_type = self._parse_str_enum(self.KEY_SPINUP_TYPE,
                                           SpinupType,
                                           SpinupType.NONE)
        if spinup_type is SpinupType.NONE:
            spinup = SpinupNone()
            self._ensure_empty(
                lambda key: f'{key} must be empty for spinup type of {spinup_type}',
                self.KEY_RPM_INITIAL,
                self.KEY_SPINUP_TIME)
        elif spinup_type is SpinupType.DEVOTION:
            rpm_initial = self._parse_float(self.KEY_RPM_INITIAL)
            spinup_time = self._parse_float(self.KEY_SPINUP_TIME)
            spinup = SpinupDevotion(rounds_per_minute_initial=rpm_initial,
                                    spinup_time_seconds=spinup_time)
        elif spinup_type is SpinupType.HAVOC:
            spinup_time = self._parse_float(self.KEY_SPINUP_TIME)
            self._ensure_empty(
                lambda key: f'{key} must be empty for spinup type of {spinup_type}',
                self.KEY_RPM_INITIAL)
            spinup = SpinupHavoc(spinup_time)
        else:
            raise ValueError(f'Spinup of {spinup_type} is unsupported.')

        return spinup

    def _parse_item(self) -> WeaponArchetype:
        # Parse basic stats
        weapon_name = self._parse_str(self.KEY_WEAPON)
        active = self._parse_bool(self.KEY_ACTIVE)
        weapon_class = self._parse_str(self.KEY_WEAPON_CLASS, default_value='')
        damage_body = self._parse_float(self.KEY_DAMAGE_BODY)
        deploy_time_secs = self._parse_float(self.KEY_DEPLOY_TIME, default_value=0)

        rpm = self._parse_rpm()
        mag = self._parse_mag()
        tactical_reload_time, full_reload_time = self._parse_tactical_and_full_reload_time()
        spinup = self._parse_spinup()

        return WeaponArchetype(name=weapon_name,
                               active=active,
                               weapon_class=weapon_class,
                               damage_body=damage_body,
                               deploy_time_secs=deploy_time_secs,
                               rounds_per_minute=rpm,
                               magazine_capacity=mag,
                               tactical_reload_time=tactical_reload_time,
                               full_reload_time=full_reload_time,
                               spinup=spinup)
