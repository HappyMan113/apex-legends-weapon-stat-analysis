import logging
import typing

from apex_stat_analysis.speech.terms import ApexTermBase


class Command:
    def __init__(self, name: ApexTermBase):
        self._name = name

    def get_name(self) -> ApexTermBase:
        return self._name

    def execute(self, arguments: typing.Iterable[ApexTermBase]) -> str:
        try:
            return self._execute(arguments)
        except Exception as ex:
            logging.exception(f'Error executing command {self._name}: {ex}')
            return 'An internal error occurred.'

    def _execute(self, arguments: typing.Iterable[ApexTermBase]) -> str:
        raise NotImplementedError(f'Must implement {self._execute.__name__} in '
                                  f'{self.__class__.__name__}.')
