import os
import threading

_SOUND_PATH = os.path.join(os.path.dirname(__file__), "..", "audio_cue", "osuhit.wav")


class RepSound:
    def __init__(self):
        self._sound  = None
        self._ready  = False
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import pygame
            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(_SOUND_PATH)
            self._ready = True
        except Exception:
            pass

    def play(self):
        if self._ready:
            try:
                self._sound.play()
            except Exception:
                pass
