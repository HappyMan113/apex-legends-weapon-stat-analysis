import abc
import json
import logging
import os
from types import MappingProxyType
from typing import Callable, Generic, Optional, Tuple, Type, TypeVar, Union, final

from apex_assistant.checker import check_str, check_type

CONFIG_VALUE = Union[str, bool]
_LOGGER = logging.getLogger()
T = TypeVar('T')
U = TypeVar('U')


class PropertyBase(abc.ABC, Generic[T]):
    def __init__(self, value: T):
        check_type(CONFIG_VALUE, optional=True, value=value)
        self._listeners: list[Callable[[T], None]] = []
        self._value = value

    @final
    def get_value(self) -> T:
        return self._value

    @final
    def _set_value(self, new_value: T):
        self._value = new_value
        self._notify_listeners(new_value=new_value)

    def add_listener(self, listener: Callable[[T], None]):
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[T], None]):
        self._listeners.remove(listener)

    def _notify_listeners(self, new_value: T):
        for listener in self._listeners:
            listener(new_value)


class Property(Generic[T], PropertyBase[T]):
    def __init__(self, name: str, value: T):
        PropertyBase.__init__(self, value=value)
        check_str(allow_blank=False, name=name)
        self._name = name

    def map(self, transformer: Callable[[T], U]) -> 'PropertyBase[U]':
        return _MappedProperty(self, transformer)

    def set_value(self, new_value: T) -> T:
        old_value = self.get_value()
        self._set_value(new_value)
        return old_value

    def get_name(self) -> str:
        return self._name


class _MappedProperty(Generic[T, U], PropertyBase[U]):
    def __init__(self, parent_property: Property[T], transformer: Callable[[T], U]):
        PropertyBase.__init__(self, transformer(parent_property.get_value()))
        self._parent_property = parent_property
        self._listener = lambda new_value: self._set_value(transformer(new_value))
        parent_property.add_listener(self._listener)

    def __del__(self):
        self._parent_property.remove_listener(self._listener)


class Config(abc.ABC):
    _REQUIRED_EXTENSION = '.json'

    def __init__(self, *properties: Property[CONFIG_VALUE]):
        self._properties = properties
        self._config_filename: str | None = None
        for prop in properties:
            prop.add_listener(self.on_value_change)

    # noinspection PyUnusedLocal
    def on_value_change(self, new_value):
        config_filename = self._config_filename
        if config_filename is not None:
            self.save(config_filename)

    @classmethod
    @final
    def load(cls: Type['Config'], config_filename: str) -> T:
        _, ext = os.path.splitext(config_filename)
        if ext != cls._REQUIRED_EXTENSION:
            configuration: dict[str, CONFIG_VALUE] = {}

        elif not os.path.exists(config_filename):
            _LOGGER.warning(f'Configuration file {config_filename} did not contain a JSON '
                            'dictionary. Content will be ignored.')
            configuration: dict[str, CONFIG_VALUE] = {}

        else:
            with open(config_filename, 'r') as fp:
                configuration: dict[str, CONFIG_VALUE] = json.load(fp)
            if not isinstance(configuration, dict):
                _LOGGER.warning(f'Configuration file {config_filename} did not contain a JSON '
                                'dictionary. Content will be ignored.')

        configuration: MappingProxyType[str, CONFIG_VALUE] = MappingProxyType(configuration)
        config = cls._load_config(configuration)
        config._config_filename = config_filename
        return config

    @classmethod
    @abc.abstractmethod
    def _load_config(cls: Type[T], configuration: MappingProxyType[str, CONFIG_VALUE]) -> T:
        raise NotImplementedError('Must implement.')

    @staticmethod
    def _get_str(configuration: MappingProxyType[str, CONFIG_VALUE],
                 key: str,
                 default_value: str | None) -> Property[str | None]:
        return Config._get_val_of_type(configuration=configuration,
                                       key=key,
                                       default_value=default_value,
                                       value_type=str)

    @staticmethod
    def _get_bool(configuration: MappingProxyType[str, CONFIG_VALUE],
                  key: str,
                  default_value: bool) -> Property[bool]:
        if default_value is None:
            raise ValueError('default_value cannot be None')
        return Config._get_val_of_type(configuration=configuration,
                                       key=key,
                                       default_value=default_value,
                                       value_type=bool)

    @staticmethod
    def _get_val_of_type(configuration: MappingProxyType[str, CONFIG_VALUE],
                         key: str,
                         default_value: T | None,
                         value_type: Type[T] | Tuple[Type[T]],
                         optional: bool = True) -> Property[T] | Property[T | None]:
        if not optional and default_value is None:
            raise ValueError('default_value cannot be None!')
        elif default_value is not None and not isinstance(default_value, value_type):
            raise TypeError(f'default_value must be an instance of {value_type}.')

        value: T | None = configuration.get(key, None)
        if value is None:
            _LOGGER.debug(f'No value found for {key}. Loading default: {default_value}.')
            value = default_value
        elif not isinstance(value, value_type):
            _LOGGER.warning(f'Value for {key} ({value}) was not of type {value_type}. Loading '
                            f'default: {default_value}.')
            value = default_value
        return Property(name=key, value=value)

    @final
    def _serialize(self) -> dict[str, Optional[CONFIG_VALUE]]:
        return {prop.get_name(): prop.get_value()
                for prop in self._properties}

    @final
    def save(self, config_filename: str) -> None:
        config: dict[str, CONFIG_VALUE] = {key: value
                                           for key, value in self._serialize().items()
                                           if value is not None}

        with open(config_filename, 'w') as fp:
            json.dump(config, fp)
