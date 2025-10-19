import logging
from threading import Event, Lock, Thread
from typing import Callable, Iterable, Sequence

from .handler import CameraHandler

PresetCallback = Callable[[int, CameraHandler], None]


class PTZPresetPatrol:
    """Executa um tour simples percorrendo presets PTZ e disparando callbacks."""

    def __init__(
        self,
        handler: CameraHandler,
        *,
        start_preset: int = 0,
        count_preset: int = 0,
        end_preset: int = 0,
        preset_timeout: float = 20.0,
        skip_presets: Iterable[int] | None = None,
        callbacks: Sequence[PresetCallback] | None = None,
    ):
        self._handler = handler
        self.start_preset = int(start_preset)
        self.count_preset = max(0, int(count_preset))
        self.end_preset = max(0, int(end_preset))
        self.preset_timeout = max(0.1, float(preset_timeout))
        self._skip_presets = {int(p) for p in (skip_presets or [])}
        self._callbacks: list[PresetCallback] = list(callbacks or [])

        self._thread: Thread | None = None
        self._stop = Event()
        self._lock = Lock()
        self._current_preset = self.start_preset
        self._last_preset: int | None = None

    # ------------------------------------------------------------------
    # Configuração

    def set_start_preset(self, value: int) -> None:
        with self._lock:
            self.start_preset = int(value)
            self._current_preset = self.start_preset

    def set_count_preset(self, value: int) -> None:
        with self._lock:
            self.count_preset = max(0, int(value))

    def set_end_preset(self, value: int) -> None:
        with self._lock:
            self.end_preset = max(0, int(value))

    def set_preset_timeout(self, value: float) -> None:
        with self._lock:
            self.preset_timeout = max(0.1, float(value))

    def add_callback(self, callback: PresetCallback) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: PresetCallback) -> None:
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    @property
    def last_preset(self) -> int | None:
        return self._last_preset

    # ------------------------------------------------------------------
    # Execução

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if not self._has_presets():
                logging.warning("Preset patrol sem presets configurados; start ignorado")
                return
            self._stop.clear()
            self._current_preset = self.start_preset
            self._thread = Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def step(self) -> bool:
        """Executa um passo do tour (útil para testes)."""
        if not self._has_presets():
            return False
        with self._lock:
            preset = self._current_preset
            self._current_preset = self._next_preset(preset)
        return self._visit_preset(preset)

    # ------------------------------------------------------------------
    # Internos

    def _has_presets(self) -> bool:
        return self.count_preset > 0 or self.end_preset > 0

    def _effective_end(self) -> int:
        if self.end_preset > 0 and self.end_preset > self.start_preset:
            return self.end_preset
        if self.count_preset > 0:
            return self.start_preset + max(1, self.count_preset)
        return self.start_preset + 1

    def _next_preset(self, current: int) -> int:
        end = self._effective_end()
        if current < self.start_preset or current >= end:
            return self.start_preset
        nxt = current + 1
        if nxt >= end:
            nxt = self.start_preset
        return nxt

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            visited = self.step()
            if not visited:
                self._handler._sleep_interruptible(0.5)

    def _visit_preset(self, preset: int) -> bool:
        if preset in self._skip_presets:
            return False
        moved = self._handler.goto_preset(preset)
        if not moved:
            return False

        half = max(self.preset_timeout / 2.0, 0.0)
        if half:
            self._handler._sleep_interruptible(half)

        for callback in list(self._callbacks):
            try:
                callback(preset, self._handler)
            except Exception:
                logging.exception("Falha ao executar callback de preset %s", preset)

        if half:
            self._handler._sleep_interruptible(half)

        self._last_preset = preset
        return True
