import abc
import logging

from apex_stat_analysis.checker import check_type
from apex_stat_analysis.speech.term import RequiredTerm, Words


class Command(abc.ABC):
    def __init__(self, term: RequiredTerm):
        check_type(RequiredTerm, term=term)
        self._term = term

    def get_term(self) -> RequiredTerm:
        return self._term

    def execute(self, arguments: Words) -> str:
        check_type(Words, arguments=arguments)
        try:
            return self._execute(arguments)
        except Exception as ex:
            logging.exception(f'Error executing command {self._term}: {ex}', exc_info=ex)
            return 'An internal error occurred.'

    @abc.abstractmethod
    def _execute(self, arguments: Words) -> str:
        raise NotImplementedError(f'Must implement {self._execute.__name__} in '
                                  f'{self.__class__.__name__}.')

    def __repr__(self):
        return repr(self._term)

    def __str__(self):
        return str(self._term)
