import logging
import os
import os.path
import re
from threading import Lock
from time import sleep
from typing import Callable, TypeVar

from huggingface_hub.file_download import are_symlinks_supported
from torch.hub import get_dir

from apex_assistant.speech.command_registry import CommandRegistry
from apex_assistant.speech.term import Words

# Logger for this module.
logger = logging.getLogger()

T = TypeVar('T')
OPENAI_API_KEY_KEY = 'OPENAI_API_KEY'
ELEVENLABS_API_KEY_KEY = 'ELEVENLABS_API_KEY'
ELEVENLABS_VOICE_NAME_KEY = 'ELEVENLABS_VOICE_NAME'
ELEVENLABS_VOICE_ID_KEY = 'ELEVENLABS_VOICE_ID'


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


def ensure_trusted_repo():
    filename = os.path.join(get_dir(), 'trusted_list')
    owner_name = 'snakers4_silero-vad'
    if not os.path.exists(filename):
        trusted_repos = tuple()
    else:
        with open(filename, 'r') as fp:
            trusted_repos = tuple(line.strip() for line in fp)

    if owner_name not in trusted_repos:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w+') as fp:
            fp.writelines(trusted_repos + (owner_name,))
        logger.info(f'Added {owner_name} to trusted repos list.')


@_preserve_logging
def create_audio_to_text_recorder():
    # Import locally because the import itself takes a long time.
    from RealtimeSTT import AudioToTextRecorder

    # Decide which STT model to use based on whether symbolic links are supported (i.e. whether
    # developer mode is enabled).
    if not are_symlinks_supported():
        logger.warning('Developer mode is not enabled. Using large-v3 Speech-to-Text model instead '
                       'of the faster distil-large-v3 model.')
        model = 'large-v3'
    else:
        model = 'distil-large-v3'
        ensure_trusted_repo()

    try:
        recorder = AudioToTextRecorder(spinner=False,
                                       device='cuda',
                                       language='en',
                                       model=model,
                                       post_speech_silence_duration=1,
                                       level=logging.WARNING)
    except PermissionError as ex:
        raise RuntimeError(
            'Must run this script from a working directory in which you have write access.') from ex

    return recorder


@_preserve_logging
def create_text_to_audio_stream():
    # Import locally because the import itself takes a long time.
    from RealtimeTTS import (OpenAIEngine, SystemEngine, ElevenlabsEngine, TextToAudioStream,
                             BaseEngine)

    engines: list[BaseEngine] = []

    elevenlabs_api_key = os.environ.get(ELEVENLABS_API_KEY_KEY)
    if isinstance(elevenlabs_api_key, str):
        voice_and_id_args: dict[str, str] = {}
        voice_name = os.environ.get(ELEVENLABS_VOICE_NAME_KEY)
        if isinstance(voice_name, str):
            voice_and_id_args['voice'] = voice_name
        voice_id = os.environ.get(ELEVENLABS_VOICE_ID_KEY)
        if isinstance(voice_id, str):
            voice_and_id_args['id'] = voice_id
        engines.append(ElevenlabsEngine(api_key=elevenlabs_api_key,
                                        model='eleven_multilingual_v2',
                                        **voice_and_id_args))

    if isinstance(os.environ.get(OPENAI_API_KEY_KEY), str):
        engines.append(OpenAIEngine(voice='echo'))

    engines.append(SystemEngine())

    return TextToAudioStream(engines, level=logging.WARNING)


class SpeechClient:
    def __init__(self, command_registry: CommandRegistry):
        if not (isinstance(os.environ.get('OPENAI_API_KEY'), str) or
                isinstance(os.environ.get(ELEVENLABS_API_KEY_KEY), str)):
            logger.warning(
                f'Neither {OPENAI_API_KEY_KEY} nor {ELEVENLABS_API_KEY_KEY} environment variables '
                'were set. The system text-to-speech engine will be used.')
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
        message = self.command_registry.process_command(words)
        self.text_to_speech(message)

    def text_to_speech(self, text: str):
        if len(text) == 0:
            return
        text = re.sub('[()]', '', text)
        if not text.endswith(('.', '!', '?')):
            text = f'{text}.'
        text = text.capitalize()
        logger.debug(f'Saying: {text}')
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
