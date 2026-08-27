import sys
import subprocess
import threading

_NO_WINDOW = 0x08000000  # Windows CREATE_NO_WINDOW

# Inline script run in a fresh subprocess per utterance.
# A fresh process guarantees a clean SAPI5 COM state — avoids the
# runAndWait() hang that occurs on repeated calls within the same process.
_SPEAK_SCRIPT = (
    "import sys, pyttsx3; "
    "e = pyttsx3.init(); "
    "e.setProperty('rate', 190); "
    "e.say(sys.argv[1]); "
    "e.runAndWait()"
)


class TTSCoach:
    """Non-blocking TTS coach.

    speak() uses a latest-message pattern: a new cue overwrites any pending
    (not yet spoken) one so the engine never falls behind mid-session.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._pending = None
        self._proc    = None
        self._event   = threading.Event()
        self._thread  = threading.Thread(target=self._manager, daemon=True)
        self._thread.start()

    def _manager(self):
        while True:
            if self._event.wait(timeout=0.5):
                self._event.clear()
                with self._lock:
                    text          = self._pending
                    self._pending = None
                if text:
                    try:
                        proc = subprocess.Popen(
                            [sys.executable, "-c", _SPEAK_SCRIPT, text],
                            creationflags=_NO_WINDOW,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        with self._lock:
                            self._proc = proc
                        proc.wait()
                        with self._lock:
                            self._proc = None
                    except Exception:
                        pass

    def speak(self, text: str):
        """Replace any pending (not yet spoken) cue with text."""
        with self._lock:
            self._pending = text
        self._event.set()

    def speak_urgent(self, text: str):
        """Interrupt any current speech and speak text immediately."""
        with self._lock:
            self._pending = text
            if self._proc is not None:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._event.set()
