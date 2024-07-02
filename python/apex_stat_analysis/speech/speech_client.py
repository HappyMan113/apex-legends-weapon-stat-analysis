import logging
import os.path
import os
from threading import Lock
from time import sleep

from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import OpenAIEngine, SystemEngine, TextToAudioStream

from apex_stat_analysis.speech.command import Command
from apex_stat_analysis.speech.command_registry import CommandRegistry
from apex_stat_analysis.speech.term_translator import ApexTranslator
from apex_stat_analysis.speech.terms import ApexTermBase, ApexTerms

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
                                            post_speech_silence_duration=2)
        self.stream = TextToAudioStream([openai_engine, system_engine])
        self._closed = False
        self.apex_translator = ApexTranslator()

    def start(self):
        self.text_to_speech('listening')
        try:
            while not self._closed:
                text = self.recorder.text()
                self.process_text(text)
        finally:
            self.stop()

    def process_text(self, text: str):
        logger.debug(f'Heard: {text}')
        terms: tuple[ApexTermBase] = self.apex_translator.translate_terms(text)
        logger.debug(f'Terms: {terms}')
        if terms == (ApexTerms.STOP_TERM,):
            self.text_to_speech('stopping')
            self.stop()
        elif len(terms) > 0:
            self._process_command(terms)

    def _process_command(self, terms: tuple[ApexTermBase]):
        assert len(terms) > 0
        result = self.find_command(terms)
        if result is not None:
            logger.debug(f'Issuing command: {terms}')
            command, idx = result
            message = command.execute(terms[:idx] + terms[idx + 1:])
            self.text_to_speech(message)
        else:
            logger.debug('Unknown command.')

    @staticmethod
    def find_command(terms: tuple[ApexTermBase]) -> tuple[Command, int] | None:
        for idx, term in enumerate(terms):
            command = CommandRegistry.get_instance().get_command(term)
            if command is not None:
                return command, idx

        return None

    def text_to_speech(self, text: str):
        if not text.endswith('.'):
            text = f'{text}.'
        logger.info(f'Saying: {text}')
        self.stream.feed(text).play()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        if self._closed:
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
