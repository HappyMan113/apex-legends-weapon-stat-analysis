from threading import Lock

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.terms import ApexTermBase

class CommandRegistry:
    INSTANCE: 'CommandRegistry | None' = None
    LOCK = Lock()

    def __init__(self):
        self._registry: dict[ApexTermBase, Command] = {}

    def register_command(self, command: Command):
        name = command.get_name()
        if name in self._registry:
            raise RuntimeError(f'Another command is registered as "{name}".')
        self._registry[name] = command

    def get_command(self, name_or_alias: ApexTermBase) -> Command | None:
        return self._registry.get(name_or_alias, None)

    @staticmethod
    def get_instance() -> 'CommandRegistry':
        with CommandRegistry.LOCK:
            if CommandRegistry.INSTANCE is None:
                CommandRegistry.INSTANCE = CommandRegistry()

        return CommandRegistry.INSTANCE
