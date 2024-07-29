import logging
from types import MappingProxyType
from typing import Type

from apex_assistant.speech.config import Config, _CONFIG_VALUE

LOGGER = logging.getLogger()


class ApexConfig(Config):
    # TODO: Maybe support configuration of opponent skill level?

    @classmethod
    def _load_config(cls: Type['ApexConfig'],
                     configuration: MappingProxyType[str, _CONFIG_VALUE]) -> 'ApexConfig':
        return ApexConfig()
