import logging
from types import MappingProxyType
from typing import Optional, Type

from apex_assistant.legend import Legend
from apex_assistant.speech.config import CONFIG_VALUE, Config, Property

LOGGER = logging.getLogger()


class ApexConfig(Config):
    # TODO: Maybe support configuration of opponent skill level?
    def __init__(self, legend_name: Property[str | None]):
        self._legend_name = legend_name
        self._legend_prop = legend_name.map(self._str_to_legend)
        super().__init__(legend_name)

    @staticmethod
    def _str_to_legend(legend_name: Optional[str]) -> Optional[Legend]:
        try:
            return Legend(legend_name)
        except ValueError:
            return None

    @staticmethod
    def _legend_to_str(legend: Optional[Legend]) -> Optional[str]:
        return str(legend) if legend is not None else None

    @classmethod
    def _load_config(cls: Type['ApexConfig'],
                     configuration: MappingProxyType[str, CONFIG_VALUE]) -> 'ApexConfig':
        legend_name = cls._get_str(configuration, key='legend', default_value=None)
        return ApexConfig(legend_name=legend_name)

    def set_legend(self, legend: Optional[Legend]) -> Optional[Legend]:
        prev_legend = self.get_legend()
        self._legend_name.set_value(self._legend_to_str(legend))
        return prev_legend

    def get_legend(self) -> Optional[Legend]:
        return self._legend_prop.get_value()
