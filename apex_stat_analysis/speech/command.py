import logging

from apex_stat_analysis.speech.term import _TermBase, Words


class Command:
    def __init__(self, name: _TermBase):
        self._name = name

    def get_name(self) -> _TermBase:
        return self._name

    def execute(self, arguments: Words) -> str:
        try:
            return self._execute(arguments)
        except Exception as ex:
            logging.exception(f'Error executing command {self._name}: {ex}', exc_info=ex)
            return 'An internal error occurred.'

    def _execute(self, arguments: Words) -> str:
        raise NotImplementedError(f'Must implement {self._execute.__name__} in '
                                  f'{self.__class__.__name__}.')

    def __repr__(self):
        return str(self._name)
