import logging
from types import MappingProxyType
from typing import Type

from apex_assistant.speech.config import Config, Property, _CONFIG_VALUE

LOGGER = logging.getLogger()


class ApexConfig(Config):
    def __init__(self,
                 default_sidearm_name: Property[str | None],
                 reload_by_default: Property[bool]):
        self._default_sidearm_name = default_sidearm_name
        self._reload_by_default = reload_by_default
        super().__init__(default_sidearm_name, reload_by_default)

    @classmethod
    def _load_config(cls: Type['ApexConfig'],
                     configuration: MappingProxyType[str, _CONFIG_VALUE]) -> 'ApexConfig':
        default_sidearm_name = cls._get_str(configuration,
                                            key='default_sidearm',
                                            default_value=None)
        reload_by_default = cls._get_bool(configuration,
                                          key='reloads',
                                          default_value=True)

        return ApexConfig(default_sidearm_name=default_sidearm_name,
                          reload_by_default=reload_by_default)

    def get_default_sidearm_name(self) -> Property[str | None]:
        return self._default_sidearm_name

    def get_reload_default(self) -> Property[bool]:
        return self._reload_by_default
