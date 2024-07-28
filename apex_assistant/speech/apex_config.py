import logging
from types import MappingProxyType
from typing import Type

from apex_assistant.speech.config import Config, Property, _CONFIG_VALUE

LOGGER = logging.getLogger()


class ApexConfig(Config):
    def __init__(self, default_sidearm_name: Property[str | None]):
        self._default_sidearm_name = default_sidearm_name
        super().__init__(default_sidearm_name)

    @classmethod
    def _load_config(cls: Type['ApexConfig'],
                     configuration: MappingProxyType[str, _CONFIG_VALUE]) -> 'ApexConfig':
        default_sidearm_name = cls._get_str(configuration,
                                            key='default_sidearm',
                                            default_value=None)

        return ApexConfig(default_sidearm_name=default_sidearm_name)

    def get_default_sidearm_name(self) -> Property[str | None]:
        return self._default_sidearm_name
