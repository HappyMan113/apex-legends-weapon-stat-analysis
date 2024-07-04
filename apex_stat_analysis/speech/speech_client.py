import logging
import os.path
import os
from threading import Lock
from time import sleep

from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import OpenAIEngine, SystemEngine, TextToAudioStream

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.command_registry import CommandRegistry
from apex_stat_analysis.speech.terms import Words

# Logger for this module.
logger = logging.getLogger()


class SpeechClient:
    def __init__(self):
        assert isinstance(os.environ.get('OPENAI_API_KEY'), str)
        openai_engine = OpenAIEngine()
        system_engine = SystemEngine()
        self._lock = Lock()
        self._sayings: list[str] = []

        self.recorder = AudioToTextRecorder(spinner=False,
                                            device='cuda',
                                            language='en',
                                            model='large-v2',
                                            post_speech_silence_duration=1)
        self.stream = TextToAudioStream([openai_engine, system_engine])
        self.command_registry = CommandRegistry.get_instance()
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

    def _process_command(self, command: Command, arguments: Words | None):
        args_str = f' {arguments}' if arguments is not None else ''
        logger.info(f'Issuing command: {command}{args_str}')
        message = command.execute(arguments)
        self.text_to_speech(message)

    def text_to_speech(self, text: str):
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
