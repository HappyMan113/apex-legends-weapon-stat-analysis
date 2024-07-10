import logging
import os
import os.path
import re
from threading import Lock
from time import sleep
from typing import Callable, TypeVar

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.command_registry import CommandRegistry
from apex_stat_analysis.speech.term import Words

# Logger for this module.
logger = logging.getLogger()


T = TypeVar('T')


def _preserve_logging[T](func: Callable[[], 'T']) -> Callable[[], T]:
    def wrapped():
        # Keep track of what old handlers we had, and then temporarily remove them to avoid
        # redundant log messages during AudioToTextRecorder initialization.
        old_handlers = tuple(logger.handlers)
        existing_handler_classes = tuple(handler.__class__ for handler in old_handlers)
        logger.handlers.clear()
        log_level = logger.getEffectiveLevel()

        result = func()

        # Add the old handlers back, and remove any of the new handlers that are redundant.
        new_handlers = tuple(logger.handlers)
        logger.handlers.extend(old_handlers)
        logger.setLevel(log_level)

        for handler in new_handlers:
            if isinstance(handler, existing_handler_classes):
                logger.removeHandler(handler)

        return result
    return wrapped


@_preserve_logging
def create_audio_to_text_recorder():
    # Import locally because the import itself takes a long time.
    from RealtimeSTT import AudioToTextRecorder
    recorder = AudioToTextRecorder(spinner=False,
                                   device='cuda',
                                   language='en',
                                   model='large-v2',
                                   post_speech_silence_duration=1,
                                   level=logging.WARNING)
    return recorder


@_preserve_logging
def create_text_to_audio_stream():
    # Import locally because the import itself takes a long time.
    from RealtimeTTS import OpenAIEngine, SystemEngine, TextToAudioStream
    openai_engine = OpenAIEngine(voice='echo')
    system_engine = SystemEngine()
    return TextToAudioStream([openai_engine, system_engine], level=logging.WARNING)


class SpeechClient:
    def __init__(self, command_registry: CommandRegistry):
        assert isinstance(os.environ.get('OPENAI_API_KEY'), str)
        self._lock = Lock()
        self._sayings: list[str] = []

        logger.info('Initializing speech-to-text recorder...')
        self.recorder = create_audio_to_text_recorder()

        logger.info('Initializing text-to-speech player...')
        self.stream = create_text_to_audio_stream()

        logger.info('STT and TTS initialized.')

        self.command_registry = command_registry
        self._closed = False

    def start(self):
        self.text_to_speech('listening')
        try:
            while not self._closed:
                text: str = self.recorder.text()
                try:
                    words = Words(text)
                    self.process_text(words)
                except ValueError:
                    # It's okay.
                    pass
        finally:
            self.stop()

    def process_text(self, words: Words):
        logger.info(f'Heard: {words}')
        for parsed in self.command_registry.get_commands(words):
            command = parsed.get_parsed()
            args = parsed.get_following_words()
            self._process_command(command, args)

    def _process_command(self, command: Command, arguments: Words):
        assert isinstance(arguments, Words)
        args_str = f' {arguments}' if len(arguments) > 0 else ''
        logger.info(f'Issuing command: {command}{args_str}')
        message = command.execute(arguments)
        self.text_to_speech(message)

    def text_to_speech(self, text: str):
        text = re.sub('[()]', '', text)
        if not text.endswith(('.', '!', '?')):
            text = f'{text}.'
        logger.info(f'Saying: {text}')
        self.stream.feed(text).play()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        if not hasattr(self, '_closed') or self._closed:
            return
        self._closed = True

        # Shutting down the recorder causes hanging. Skip for now.
        self.recorder = None
        # print('Shutting down...')
        # self.recorder.shutdown()

        logger.debug('Waiting...')
        while self.stream.is_playing():
            sleep(0.1)
        logger.debug('Stopping...')
        self.stream.stop()
        logger.debug('Stopped.')

    def __del__(self):
        self.stop()
