# -*- coding: utf-8 -*-
"""
Тиргум — десктопный переводчик RU ⇄ EN с автоматической озвучкой.

Как это работает:
    1. Направление перевода (RU → EN или EN → RU) выбирается кнопкой ⇄
       в шапке окна или клавишей F4.
    2. Пользователь нажимает кнопку микрофона (или F2) и говорит на языке-
       источнике — речь записывается и распознаётся (SpeechRecognition,
       Google Web Speech API); распознанный текст попадает в поле ввода
       и сразу отправляется на перевод.
       Либо вводит/вставляет текст вручную и нажимает
       «Перевести» или Enter (Shift+Enter — перенос строки).
    3. В фоновом потоке (QThread) текст переводится через deep-translator
       (GoogleTranslator), затем gTTS синтезирует речь на языке перевода в mp3.
    4. Готовый перевод показывается крупным шрифтом, а mp3 сразу
       воспроизводится через pygame.mixer.
    5. Кнопка 🔊 («Повторить озвучку») проигрывает уже готовый mp3 заново —
       без повторного перевода и без повторного обращения к сети.

Работа с временными файлами:
    - Все mp3 кладутся в собственную временную папку приложения
      (создаётся через tempfile.mkdtemp).
    - Каждый новый файл получает уникальное имя (uuid), поэтому мы никогда
      не пытаемся перезаписать файл, который pygame держит открытым
      (типичная ошибка «PermissionError: [WinError 32]» на Windows).
    - mp3 целиком загружается в память (pygame.mixer.Sound), поэтому файл
      не остаётся открытым и безопасно удаляется при следующем переводе.
    - При закрытии приложения временная папка удаляется целиком.

Запуск:
    pip install -r requirements.txt
    python tirgum.py
"""

import os
import sys
import shutil
import tempfile
import uuid

# Скрываем приветственное сообщение pygame в консоли (должно быть ДО импорта pygame).
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import speech_recognition as sr
from deep_translator import GoogleTranslator, MyMemoryTranslator
from gtts import gTTS

from PyQt6.QtCore import QRectF, QSettings, Qt, QThread, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QRadialGradient,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

# Поддерживаемые языки. Направление перевода (RU → EN или EN → RU)
# переключается кнопкой ⇄ в шапке окна или клавишей F4.
LANGUAGES = {
    "ru": {
        "code": "RU",
        "name": "Русский",
        "say": "по-русски",
        "google": "ru",       # код для GoogleTranslator
        "mymemory": "ru-RU",  # резервный MyMemoryTranslator требует коды с регионом
        "tts": "ru",          # язык озвучки gTTS
        "speech": "ru-RU",    # язык распознавания речи
    },
    "en": {
        "code": "EN",
        "name": "Английский",
        "say": "по-английски",
        "google": "en",
        "mymemory": "en-GB",
        "tts": "en",
        "speech": "en-US",
    },
}
DEFAULT_DIRECTION = ("ru", "en")  # (язык источника, язык перевода) при запуске

MAX_CHARS = 5000          # лимит GoogleTranslator на один запрос

# Голосовой ввод (распознавание через бесплатный Google Web Speech API)
LISTEN_TIMEOUT = 7       # сколько секунд ждать начала речи
PHRASE_LIMIT = 15         # максимальная длина одной фразы, секунд
PAUSE_THRESHOLD = 1.0     # пауза (сек), после которой фраза считается законченной

# Темп озвучки перевода: (подпись кнопки, подсказка, множитель скорости).
# Замедление делается без изменения высоты голоса — см. time_stretch().
SPEEDS = [
    ("1×", "Обычная скорость", 1.0),
    ("0.85×", "Немного медленнее", 0.85),
    ("0.7×", "Медленно", 0.7),
]
DEFAULT_SPEED = 0.85
MIXER_FREQ = 44100        # частота дискретизации воспроизведения


# ---------------------------------------------------------------------------
# Замедление речи без изменения высоты голоса (алгоритм WSOLA)
# ---------------------------------------------------------------------------

def time_stretch(samples, speed: float, sample_rate: int):
    """
    Меняет темп звука, сохраняя высоту голоса (WSOLA — Waveform Similarity
    Overlap-Add). speed < 1 — медленнее, speed = 1 — без изменений.

    Звук режется на короткие перекрывающиеся кадры (40 мс). Кадры берутся
    из исходника с шагом hop_out * speed, а складываются с шагом hop_out —
    так запись становится длиннее. Чтобы на стыках не было «бульканья»,
    каждый следующий кадр сдвигается в пределах ±10 мс туда, где он лучше
    всего совпадает по форме волны с естественным продолжением предыдущего.

    samples — массив numpy int16 формы (кол-во сэмплов, каналы).
    """
    if abs(speed - 1.0) < 1e-3 or len(samples) == 0:
        return samples

    frame = int(0.040 * sample_rate)       # длина кадра
    hop_out = frame // 2                   # шаг в результате (перекрытие 50 %)
    hop_in = hop_out * speed               # шаг по исходнику
    tol = int(0.010 * sample_rate)         # окно поиска лучшего совпадения
    dec = 4                                # прореживание для быстрого поиска

    channels = samples.shape[1]
    x = samples.astype(np.float32)
    # Поля из тишины, чтобы поиск совпадения не выходил за края массива.
    xp = np.concatenate([
        np.zeros((tol, channels), np.float32),
        x,
        np.zeros((frame + tol + hop_out, channels), np.float32),
    ])
    mono = xp.mean(axis=1)[::dec]
    window = np.hanning(frame).astype(np.float32)

    n_frames = max(1, int((len(x) - frame) / hop_in) + 1)
    out_len = (n_frames - 1) * hop_out + frame
    out = np.zeros((out_len, channels), np.float32)
    weight = np.zeros(out_len, np.float32)

    prev = tol
    for k in range(n_frames):
        ideal = tol + int(k * hop_in)
        if k == 0:
            best = ideal
        else:
            natural = prev + hop_out       # естественное продолжение прошлого кадра
            template = mono[natural // dec:(natural + frame) // dec]
            lo = (ideal - tol) // dec
            region = mono[lo:lo + (2 * tol + frame) // dec]
            corr = np.correlate(region, template, "valid")
            best = lo * dec + int(np.argmax(corr)) * dec
        pos = k * hop_out
        out[pos:pos + frame] += xp[best:best + frame] * window[:, None]
        weight[pos:pos + frame] += window
        prev = best

    out /= np.maximum(weight, 1e-3)[:, None]
    return np.clip(out, -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Менеджер аудио: воспроизведение и безопасная работа с временными mp3
# ---------------------------------------------------------------------------

class AudioManager:
    """
    Отвечает за временную папку, воспроизведение mp3 с нужным темпом
    и уборку за собой.

    mp3 целиком декодируется в память (pygame.mixer.Sound) — после загрузки
    файл на диске больше не занят, и его можно спокойно удалить.
    Замедленные версии кэшируются, поэтому повтор срабатывает мгновенно.
    """

    def __init__(self):
        # Собственная временная папка — ничего не мусорим в общем %TEMP%.
        self.temp_dir = tempfile.mkdtemp(prefix="tirgum_")
        self.current_file = None   # путь к mp3 текущего перевода
        self.available = True      # False, если аудиоустройство недоступно
        self._samples = None       # декодированный звук текущего mp3 (int16)
        self._cache = {}           # темп → готовый pygame.mixer.Sound

        try:
            # 44.1 кГц, 16 бит, стерео — формат, с которым работает time_stretch.
            pygame.mixer.init(frequency=MIXER_FREQ, size=-16, channels=2)
            self.rate, self.bits, self.channels = pygame.mixer.get_init()
        except pygame.error as exc:
            # Нет звуковой карты / устройство занято — приложение всё равно
            # работает как переводчик, просто без звука.
            self.available = False
            self.init_error = str(exc)

    def new_file_path(self) -> str:
        """Уникальное имя для нового mp3 — исключает конфликт с открытым файлом."""
        return os.path.join(self.temp_dir, f"speech_{uuid.uuid4().hex}.mp3")

    @staticmethod
    def _safe_remove(path):
        """Удаляет файл, не падая, если он уже удалён или ещё занят."""
        if not path:
            return
        try:
            os.remove(path)
        except OSError:
            # Если файл всё же занят — он будет удалён вместе с папкой при выходе.
            pass

    def stop(self):
        """Останавливает текущее воспроизведение (звук остаётся для повтора)."""
        if self.available:
            try:
                pygame.mixer.stop()
            except pygame.error:
                pass

    def set_current(self, path: str):
        """Делает новый mp3 текущим, а предыдущий — безопасно удаляет."""
        old = self.current_file
        self.stop()
        self._samples = None
        self._cache.clear()
        self.current_file = path
        if old and old != path:
            self._safe_remove(old)

    def _sound_for(self, speed: float):
        """Возвращает звук текущего перевода в нужном темпе (с кэшированием)."""
        if speed in self._cache:
            return self._cache[speed]
        original = pygame.mixer.Sound(self.current_file)
        if abs(speed - 1.0) < 1e-3 or self.bits != -16:
            sound = original   # обычный темп или неожиданный формат микшера
        else:
            if self._samples is None:
                raw = np.frombuffer(original.get_raw(), np.int16)
                self._samples = raw.reshape(-1, self.channels)
            stretched = time_stretch(self._samples, speed, self.rate)
            sound = pygame.mixer.Sound(buffer=np.ascontiguousarray(stretched).tobytes())
        self._cache[speed] = sound
        return sound

    def play(self, speed: float = 1.0) -> bool:
        """Проигрывает текущий перевод с начала. Вызов не блокирует интерфейс."""
        if not self.available or not self.current_file:
            return False
        if speed not in self._cache and not os.path.exists(self.current_file):
            return False
        try:
            self.stop()
            self._sound_for(speed).play()
            return True
        except pygame.error:
            return False

    def cleanup(self):
        """Вызывается при выходе: освобождает микшер и удаляет временную папку."""
        self.stop()
        self._cache.clear()
        if self.available:
            try:
                pygame.mixer.quit()
            except pygame.error:
                pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Фоновый поток: перевод + синтез речи
# ---------------------------------------------------------------------------

class TranslateWorker(QThread):
    """
    Выполняет сетевые операции вне GUI-потока, чтобы окно не «зависало».

    Сигналы:
        translated(str, str)   — перевод готов + какой сервис перевёл (показываем сразу);
        audio_ready(str)       — mp3 сгенерирован, передаём путь к файлу;
        failed(str, bool)      — ошибка; второй аргумент True, если упал только синтез речи.
    """

    translated = pyqtSignal(str, str)
    audio_ready = pyqtSignal(str)
    failed = pyqtSignal(str, bool)

    def __init__(self, text: str, src: str, tgt: str, mp3_path: str, need_audio: bool,
                 parent=None):
        super().__init__(parent)
        self.text = text
        self.src = LANGUAGES[src]   # язык исходного текста
        self.tgt = LANGUAGES[tgt]   # язык перевода (и озвучки)
        self.mp3_path = mp3_path
        self.need_audio = need_audio

    def _translate(self) -> tuple[str, str]:
        """
        Переводит текст. Возвращает (перевод, название сервиса).

        Основной сервис — GoogleTranslator. Google иногда временно блокирует
        IP (ответ 429 «Too Many Requests» / страница с капчей) — тогда
        используем резервный бесплатный сервис из той же библиотеки
        deep-translator (MyMemory), чтобы приложение продолжало работать.
        """
        try:
            result = GoogleTranslator(
                source=self.src["google"], target=self.tgt["google"]
            ).translate(self.text)
            if result and result.strip():
                return result.strip(), "Google"
            google_error = "пустой ответ"
        except Exception as exc:  # сеть, лимиты, неожиданный ответ сервиса
            google_error = str(exc)

        try:
            result = MyMemoryTranslator(
                source=self.src["mymemory"], target=self.tgt["mymemory"]
            ).translate(self.text)
        except Exception as exc:
            raise RuntimeError(
                f"Google: {google_error}\nMyMemory (резерв): {exc}"
            ) from exc
        if not result or not result.strip():
            raise RuntimeError(f"Google: {google_error}\nMyMemory (резерв): пустой ответ")
        return result.strip(), "MyMemory (резерв — Google недоступен)"

    def run(self):
        # --- 1. Перевод ---
        try:
            translation, service = self._translate()
        except Exception as exc:
            self.failed.emit(f"Не удалось перевести текст:\n{exc}", False)
            return

        self.translated.emit(translation, service)

        if not self.need_audio:
            return

        # --- 2. Синтез речи в mp3 ---
        try:
            gTTS(text=translation, lang=self.tgt["tts"]).save(self.mp3_path)
        except Exception as exc:
            # Недописанный файл удаляем, чтобы не оставлять мусор.
            try:
                os.remove(self.mp3_path)
            except OSError:
                pass
            self.failed.emit(f"Перевод готов, но озвучить не удалось:\n{exc}", True)
            return

        self.audio_ready.emit(self.mp3_path)


# ---------------------------------------------------------------------------
# Фоновый поток: запись с микрофона + распознавание речи
# ---------------------------------------------------------------------------

class ListenWorker(QThread):
    """
    Слушает микрофон, пока пользователь не замолчит, и распознаёт речь
    на языке-источнике текущего направления (русский или английский).

    Сигналы:
        listening()        — шумоподавление откалибровано, можно говорить;
        recognized(str)    — распознанный текст;
        failed(str)        — понятное сообщение об ошибке.
    """

    listening = pyqtSignal()
    recognized = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, lang: str, parent=None):
        super().__init__(parent)
        self.speech_lang = LANGUAGES[lang]["speech"]

    def run(self):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = PAUSE_THRESHOLD
        recognizer.dynamic_energy_threshold = True

        # --- 1. Запись ---
        try:
            with sr.Microphone() as source:
                # Полсекунды слушаем фон, чтобы отличать речь от шума.
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.listening.emit()
                audio = recognizer.listen(
                    source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_LIMIT
                )
        except sr.WaitTimeoutError:
            self.failed.emit("Не услышал речь — нажмите 🎤 и говорите сразу.")
            return
        except (OSError, AttributeError) as exc:
            # OSError — нет микрофона или доступ запрещён в настройках Windows;
            # AttributeError — не установлен PyAudio.
            self.failed.emit(
                "Микрофон недоступен. Проверьте, что он подключён и что в "
                "«Параметры → Конфиденциальность → Микрофон» разрешён доступ "
                f"для классических приложений.\n({exc})"
            )
            return

        # --- 2. Распознавание ---
        try:
            text = recognizer.recognize_google(audio, language=self.speech_lang)
        except sr.UnknownValueError:
            self.failed.emit("Не удалось разобрать речь — попробуйте сказать чётче.")
            return
        except sr.RequestError as exc:
            self.failed.emit(f"Сервис распознавания речи недоступен (нужен интернет).\n({exc})")
            return

        text = (text or "").strip()
        if text:
            self.recognized.emit(text)
        else:
            self.failed.emit("Не удалось разобрать речь — попробуйте ещё раз.")


# ---------------------------------------------------------------------------
# Поле ввода: Enter — отправка, Shift+Enter — новая строка
# ---------------------------------------------------------------------------

class InputEdit(QPlainTextEdit):
    submitted = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent):
        is_enter = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        if is_enter and not shift:
            self.submitted.emit()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Оформление: палитра и иконки
# ---------------------------------------------------------------------------

# Иконки берём из системного шрифта Windows (Segoe Fluent Icons на Windows 11,
# Segoe MDL2 Assets на Windows 10) — они чёткие при любом масштабе экрана.
ICON_FONTS = ["Segoe Fluent Icons", "Segoe MDL2 Assets"]
ICON_MIC = ""
ICON_VOLUME = ""
ICON_COPY = ""
ICON_SWAP = ""

UI_FONT = "Segoe UI"

# Цвет точки-индикатора в строке статуса для каждого состояния.
STATUS_COLORS = {
    "idle": "#6b7194",     # ожидание
    "busy": "#f5b544",     # идёт работа
    "listen": "#ff5f7e",   # запись с микрофона
    "ok": "#3ddc97",       # успех
    "error": "#ff6b6b",    # ошибка
}


def icon_font(size: int) -> QFont:
    font = QFont()
    font.setFamilies(ICON_FONTS)
    font.setPointSize(size)
    return font


# ---------------------------------------------------------------------------
# Круглая кнопка микрофона с анимацией «волн» во время записи
# ---------------------------------------------------------------------------

class MicButton(QAbstractButton):
    """
    Рисуется вручную через QPainter.
    Состояния: "idle" — готова, "preparing" — калибровка микрофона,
    "listening" — идёт запись (вокруг кнопки расходятся волны).
    """

    RADIUS = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(140, 140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.state = "idle"
        self._hover = False
        self._phase = 0.0

        # Бесконечная анимация фазы волн (0 → 1 за 1.4 с).
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1400)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_phase)

    def set_state(self, state: str):
        self.state = state
        if state == "listening":
            self._anim.start()
        else:
            self._anim.stop()
            self._phase = 0.0
        self.update()

    def _on_phase(self, value):
        self._phase = float(value)
        self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def hitButton(self, pos):
        # Кликабелен только сам круг, а не пустые углы виджета.
        center = QRectF(self.rect()).center()
        dx, dy = pos.x() - center.x(), pos.y() - center.y()
        return dx * dx + dy * dy <= (self.RADIUS + 4) ** 2

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        center = QRectF(self.rect()).center()
        r = self.RADIUS

        # Цвета круга в зависимости от состояния.
        if self.state == "preparing":
            top, bottom = QColor("#8d92b5"), QColor("#5b6088")
        elif self.state == "listening" or self.isEnabled():
            top, bottom = QColor("#ff7a8a"), QColor("#e0306a")
            if self._hover and self.state == "idle":
                top, bottom = top.lighter(112), bottom.lighter(112)
        else:
            top, bottom = QColor("#3a3f5e"), QColor("#2c3150")

        # Волны во время записи: два кольца со сдвигом фазы.
        if self.state == "listening":
            for shift in (0.0, 0.5):
                ph = (self._phase + shift) % 1.0
                ring = QColor("#ff5f7e")
                ring.setAlpha(int(110 * (1.0 - ph)))
                p.setBrush(ring)
                rr = r + 28 * ph
                p.drawEllipse(center, rr, rr)

        # Мягкое свечение под кнопкой.
        glow = QRadialGradient(center, r + 18)
        glow_color = QColor(bottom)
        glow_color.setAlpha(90 if self.isEnabled() or self.state != "idle" else 0)
        glow.setColorAt(0.6, glow_color)
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(glow)
        p.drawEllipse(center, r + 18, r + 18)

        # Сам круг с градиентом.
        grad = QLinearGradient(center.x() - r, center.y() - r, center.x() + r, center.y() + r)
        grad.setColorAt(0.0, top)
        grad.setColorAt(1.0, bottom)
        p.setBrush(grad)
        p.drawEllipse(center, r, r)

        # Иконка микрофона.
        p.setPen(QColor("#ffffff") if self.isEnabled() or self.state != "idle" else QColor("#7d82a6"))
        p.setFont(icon_font(24))
        p.drawText(QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r),
                   Qt.AlignmentFlag.AlignCenter, ICON_MIC)
        p.end()


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Тиргум — перевод с озвучкой")
        self.resize(820, 720)
        self.setMinimumSize(620, 600)

        self.audio = AudioManager()
        self.worker = None            # текущий фоновый поток
        self.pending_mp3 = None       # mp3, который сейчас генерируется
        self.service = ""             # каким сервисом сделан текущий перевод
        self.listener = None          # поток записи с микрофона
        self.translate_after_listen = False  # речь распознана → перевести после записи
        self.status_kind = "idle"     # текущее состояние строки статуса
        self.src, self.tgt = DEFAULT_DIRECTION  # направление перевода

        # Темп озвучки запоминается между запусками (реестр Windows, QSettings).
        self.settings = QSettings("Tirgum", "Tirgum")
        saved = self.settings.value("speed", DEFAULT_SPEED, type=float)
        known = [value for _, _, value in SPEEDS]
        self.speed = saved if saved in known else DEFAULT_SPEED

        self._build_ui()
        self._apply_style()
        self._apply_direction()
        self.set_status("Готово к работе — нажмите микрофон или введите текст", "idle")

        if not self.audio.available:
            self.set_status("Аудиоустройство недоступно — перевод работает без звука", "error")

    # ----- построение интерфейса -----

    @staticmethod
    def _lang_header():
        """
        Заголовок карточки: плашка с кодом языка + название.
        Возвращает (layout, плашка, название) — текст задаётся в _apply_direction.
        """
        row = QHBoxLayout()
        row.setSpacing(10)
        badge = QLabel()
        badge.setObjectName("badge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(22)
        title = QLabel()
        title.setObjectName("cardTitle")
        row.addWidget(badge, alignment=Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(title)
        row.addStretch(1)
        return row, badge, title

    @staticmethod
    def _icon_button(glyph: str, tooltip: str) -> QPushButton:
        btn = QPushButton(glyph)
        btn.setObjectName("iconBtn")   # шрифт иконок задаётся в QSS (#iconBtn)
        btn.setFixedSize(40, 40)
        btn.setToolTip(tooltip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    def _build_speed_selector(self) -> QFrame:
        """Сегментный переключатель темпа озвучки: 1× / 0.85× / 0.7×."""
        box = QFrame()
        box.setObjectName("segBox")
        row = QHBoxLayout(box)
        row.setContentsMargins(10, 3, 3, 3)
        row.setSpacing(2)
        caption = QLabel("Темп")
        caption.setObjectName("segCaption")
        row.addWidget(caption)
        row.addSpacing(4)

        self.speed_group = QButtonGroup(self)
        self.speed_group.setExclusive(True)
        for label, tooltip, value in SPEEDS:
            btn = QPushButton(label)
            btn.setObjectName("seg")
            btn.setCheckable(True)
            btn.setChecked(value == self.speed)
            btn.setToolTip(f"{tooltip} — темп озвучки перевода")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, v=value: self.on_speed_changed(v))
            self.speed_group.addButton(btn)
            row.addWidget(btn)
        return box

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("root")
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 18)
        root.setSpacing(16)

        # --- Шапка: название, подзаголовок и направление перевода ---
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        app_title = QLabel("Тиргум")
        app_title.setObjectName("appTitle")
        self.subtitle = QLabel()
        self.subtitle.setObjectName("subtitle")
        titles.addWidget(app_title)
        titles.addWidget(self.subtitle)
        header.addLayout(titles)
        header.addStretch(1)

        # Направление перевода: плашка «RU → EN» и кнопка ⇄ — обе меняют направление.
        self.btn_direction = QPushButton()
        self.btn_direction.setObjectName("direction")
        self.btn_direction.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_direction.setToolTip("Поменять направление перевода (F4)")
        self.btn_direction.clicked.connect(self.swap_direction)
        self.btn_swap = self._icon_button(ICON_SWAP, "Поменять направление перевода (F4)")
        self.btn_swap.clicked.connect(self.swap_direction)
        QShortcut(QKeySequence("F4"), self, activated=self.swap_direction)
        header.addWidget(self.btn_direction, alignment=Qt.AlignmentFlag.AlignVCenter)
        header.addSpacing(8)
        header.addWidget(self.btn_swap, alignment=Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(header)

        # --- Карточка ввода (язык-источник) ---
        in_card = QFrame()
        in_card.setObjectName("card")
        in_layout = QVBoxLayout(in_card)
        in_layout.setContentsMargins(20, 16, 16, 14)
        in_layout.setSpacing(8)
        in_header, self.in_badge, self.in_title = self._lang_header()
        in_layout.addLayout(in_header)

        self.input = InputEdit()
        self.input.setObjectName("textIn")
        self.input.setPlaceholderText("Скажите фразу в микрофон или введите текст…")
        self.input.setFont(QFont(UI_FONT, 14))
        self.input.submitted.connect(self.on_translate)
        in_layout.addWidget(self.input, stretch=1)

        in_footer = QHBoxLayout()
        hint = QLabel("Enter — перевести  ·  Shift+Enter — новая строка")
        hint.setObjectName("hint")
        in_footer.addWidget(hint)
        in_footer.addStretch(1)
        self.btn_translate = QPushButton("Перевести")
        self.btn_translate.setObjectName("primary")
        self.btn_translate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_translate.clicked.connect(self.on_translate)
        in_footer.addWidget(self.btn_translate)
        in_layout.addLayout(in_footer)
        root.addWidget(in_card, stretch=2)

        # --- Центральная кнопка микрофона ---
        # Голосовой ввод: говорим → распознаём → переводим → озвучиваем.
        mic_box = QVBoxLayout()
        mic_box.setSpacing(0)
        self.btn_mic = MicButton()
        self.btn_mic.clicked.connect(self.on_listen)
        QShortcut(QKeySequence("F2"), self, activated=self.on_listen)
        self.mic_hint = QLabel()
        self.mic_hint.setObjectName("micHint")
        self.mic_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_mic_hint("Нажмите и говорите  ·  F2")
        mic_box.addWidget(self.btn_mic, alignment=Qt.AlignmentFlag.AlignHCenter)
        mic_box.addWidget(self.mic_hint)
        root.addLayout(mic_box)

        # --- Карточка вывода (язык перевода) ---
        out_card = QFrame()
        out_card.setObjectName("card")
        out_layout = QVBoxLayout(out_card)
        out_layout.setContentsMargins(20, 16, 16, 16)
        out_layout.setSpacing(8)

        out_header, self.out_badge, self.out_title = self._lang_header()
        self.btn_copy = self._icon_button(ICON_COPY, "Скопировать перевод")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self.on_copy)
        self.btn_repeat = self._icon_button(ICON_VOLUME, "Повторить озвучку")
        self.btn_repeat.setObjectName("speakBtn")
        self.btn_repeat.setEnabled(False)
        self.btn_repeat.clicked.connect(self.on_repeat)
        out_header.addWidget(self._build_speed_selector())
        out_header.addSpacing(6)
        out_header.addWidget(self.btn_copy)
        out_header.addWidget(self.btn_repeat)
        out_layout.addLayout(out_header)

        # Поле вывода только для чтения, но текст можно выделять и копировать.
        self.output = QTextEdit()
        self.output.setObjectName("textOut")
        self.output.setReadOnly(True)
        self.output.setFont(QFont(UI_FONT, 22, QFont.Weight.DemiBold))
        self.output.setPlaceholderText("Здесь появится перевод")
        out_layout.addWidget(self.output, stretch=1)
        root.addWidget(out_card, stretch=3)

        # --- Строка статуса с цветным индикатором ---
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.status = QLabel()
        self.status.setObjectName("status")
        status_row.addWidget(self.status_dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        status_row.addWidget(self.status, stretch=1)
        root.addLayout(status_row)

        # Цвет подсказок-плейсхолдеров задаётся через палитру (в QSS его нет).
        for edit in (self.input, self.output):
            pal = edit.palette()
            pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#5d6386"))
            edit.setPalette(pal)

        self.setCentralWidget(central)
        self.input.setFocus()

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget#root {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #0e1120, stop:1 #171b31);
            }
            QWidget { color: #e8eaf6; font-family: "Segoe UI"; }
            QToolTip {
                background: #252a45; color: #e8eaf6; border: 1px solid #353b5e;
                padding: 4px 8px; border-radius: 6px;
            }

            QLabel#appTitle { font-size: 22pt; font-weight: 700; color: #ffffff; }
            QLabel#subtitle { font-size: 10.5pt; color: #8a90b4; }
            QPushButton#direction {
                font-size: 10pt; font-weight: 700; color: #c9c6ff;
                background: rgba(124, 108, 255, 0.16);
                border: 1px solid rgba(124, 108, 255, 0.45);
                border-radius: 16px; padding: 7px 16px;
            }
            QPushButton#direction:hover { background: rgba(124, 108, 255, 0.30); color: #ffffff; }
            QPushButton#direction:disabled { color: #6b7194; border-color: #2d3252; }

            QFrame#card {
                background: #1a1e36;
                border: 1px solid #272c4b;
                border-radius: 18px;
            }
            QLabel#badge {
                font-size: 8.5pt; font-weight: 700; color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #7c6cff, stop:1 #5b8cff);
                border-radius: 8px; padding: 3px 8px; min-width: 22px;
            }
            QLabel#cardTitle { font-size: 10.5pt; font-weight: 600; color: #aab0d6; }
            QLabel#hint { font-size: 9pt; color: #5d6386; }
            QLabel#micHint { font-size: 9.5pt; color: #8a90b4; }
            QLabel#status { font-size: 9.5pt; color: #8a90b4; }

            QPlainTextEdit#textIn, QTextEdit#textOut {
                background: transparent; border: none;
                selection-background-color: #5b5bd6; selection-color: #ffffff;
            }
            QTextEdit#textOut { color: #ffffff; }

            QPushButton#primary {
                font-size: 10.5pt; font-weight: 600; color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #7c6cff, stop:1 #5b8cff);
                border: none; border-radius: 12px; padding: 9px 22px;
            }
            QPushButton#primary:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #8d7fff, stop:1 #6d9aff);
            }
            QPushButton#primary:pressed { background: #5b5bd6; }
            QPushButton#primary:disabled { background: #2d3252; color: #6b7194; }

            QFrame#segBox {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 17px;
            }
            QLabel#segCaption { font-size: 9pt; color: #6b7194; }
            QPushButton#seg {
                font-size: 9pt; font-weight: 600; color: #8a90b4;
                background: transparent; border: none; border-radius: 13px;
                padding: 5px 10px;
            }
            QPushButton#seg:hover { color: #e8eaf6; }
            QPushButton#seg:checked {
                color: #ffffff; background: rgba(124, 108, 255, 0.45);
            }

            QPushButton#iconBtn, QPushButton#speakBtn {
                font-family: "Segoe Fluent Icons", "Segoe MDL2 Assets"; font-size: 13pt;
                color: #c9cdf0; background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 20px;
            }
            QPushButton#iconBtn:hover, QPushButton#speakBtn:hover {
                background: rgba(255, 255, 255, 0.13);
            }
            QPushButton#speakBtn:enabled {
                color: #ffffff; background: rgba(124, 108, 255, 0.35);
                border: 1px solid rgba(124, 108, 255, 0.6);
            }
            QPushButton#speakBtn:enabled:hover { background: rgba(124, 108, 255, 0.55); }
            QPushButton#iconBtn:disabled, QPushButton#speakBtn:disabled {
                color: #454a6b; background: rgba(255, 255, 255, 0.03);
            }

            QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
            QScrollBar::handle:vertical { background: #353b5e; border-radius: 4px; min-height: 30px; }
            QScrollBar::handle:vertical:hover { background: #474e78; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

    # ----- вспомогательные методы интерфейса -----

    def set_status(self, text: str, kind: str = "idle"):
        """Текст статуса + цвет точки-индикатора (idle/busy/listen/ok/error)."""
        self.status_kind = kind
        self.status.setText(text)
        self.status_dot.setStyleSheet(
            f"background: {STATUS_COLORS.get(kind, STATUS_COLORS['idle'])}; border-radius: 4px;"
        )

    def _apply_direction(self):
        """Обновляет все подписи под текущее направление перевода."""
        src, tgt = LANGUAGES[self.src], LANGUAGES[self.tgt]
        self.btn_direction.setText(f"{src['code']}  →  {tgt['code']}")
        self.subtitle.setText(f"Говорите {src['say']} — слушайте {tgt['say']}")
        self.in_badge.setText(src["code"])
        self.in_title.setText(src["name"])
        self.out_badge.setText(tgt["code"])
        self.out_title.setText(tgt["name"])
        self.btn_mic.setToolTip(f"Скажите фразу {src['say']} (F2)")

    def swap_direction(self):
        """Меняет направление RU ⇄ EN; тексты в полях меняются местами."""
        if self._busy():
            return
        self.src, self.tgt = self.tgt, self.src

        # Бывший перевод становится исходным текстом — удобно перевести обратно.
        old_input = self.input.toPlainText()
        old_output = self.output.toPlainText()
        self.input.setPlainText(old_output)
        self.output.setPlainText(old_input)

        # Готовый mp3 озвучивал прежний перевод — к новому полю вывода он не относится.
        self.audio.stop()
        self.btn_repeat.setEnabled(False)
        self.btn_copy.setEnabled(bool(old_input.strip()))

        self._apply_direction()
        src, tgt = LANGUAGES[self.src], LANGUAGES[self.tgt]
        self.set_status(f"Направление: {src['name']} → {tgt['name']}", "idle")
        self.input.setFocus()

    def _set_mic_hint(self, text: str, active: bool = False):
        self.mic_hint.setText(text)
        self.mic_hint.setStyleSheet("color: #ff8fa3; font-weight: 600;" if active else "")

    # ----- обработчики -----

    def _busy(self) -> bool:
        """True, если сейчас идёт запись с микрофона или перевод."""
        return any(t is not None and t.isRunning() for t in (self.worker, self.listener))

    def on_listen(self):
        """Запускает запись с микрофона в фоновом потоке."""
        if self._busy():
            return
        # Останавливаем озвучку, чтобы микрофон не «услышал» саму программу.
        self.audio.stop()

        self.btn_mic.setEnabled(False)
        self.btn_mic.set_state("preparing")
        self._set_mic_hint("Секунду тишины — настраиваю микрофон…")
        self.btn_translate.setEnabled(False)
        self._set_direction_enabled(False)
        self.set_status("Настраиваю микрофон…", "busy")

        self.listener = ListenWorker(self.src, self)
        self.listener.listening.connect(self.on_listening)
        self.listener.recognized.connect(self.on_recognized)
        self.listener.failed.connect(self.on_listen_failed)
        self.listener.finished.connect(self.on_listener_finished)
        self.listener.start()

    def on_listening(self):
        self.btn_mic.set_state("listening")
        self._set_mic_hint(f"Слушаю — говорите {LANGUAGES[self.src]['say']}", active=True)
        self.set_status("Говорите… пауза в конце фразы — и я переведу", "listen")

    def on_recognized(self, text: str):
        """Речь распознана — показываем текст и сразу переводим."""
        self.input.setPlainText(text)
        self.set_status(f"Распознано: «{text}»", "ok")
        self.translate_after_listen = True

    def on_listen_failed(self, message: str):
        self.set_status(message.splitlines()[0], "error")

    def on_listener_finished(self):
        recognized = self.translate_after_listen
        self.translate_after_listen = False
        self.listener = None
        self.btn_mic.setEnabled(True)
        self.btn_mic.set_state("idle")
        self._set_mic_hint("Нажмите и говорите  ·  F2")
        self.btn_translate.setEnabled(True)
        self._set_direction_enabled(True)
        if recognized:
            self.on_translate()   # запускаем только после завершения потока записи

    def on_translate(self):
        """Запускает перевод и озвучку в фоновом потоке."""
        if self._busy():
            return  # защита от повторного нажатия, пока идёт предыдущий запрос

        text = self.input.toPlainText().strip()
        if not text:
            self.set_status("Сначала скажите фразу или введите текст", "error")
            return
        if len(text) > MAX_CHARS:
            QMessageBox.warning(
                self, "Слишком длинный текст",
                f"Максимум {MAX_CHARS} символов за раз (сейчас {len(text)})."
            )
            return

        self._set_busy(True)
        self.set_status("Перевожу…", "busy")

        self.pending_mp3 = self.audio.new_file_path()
        self.worker = TranslateWorker(
            text, self.src, self.tgt, self.pending_mp3, self.audio.available, self
        )
        self.worker.translated.connect(self.on_translated)
        self.worker.audio_ready.connect(self.on_audio_ready)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_translated(self, translation: str, service: str):
        """Перевод готов — показываем его сразу, озвучка догружается параллельно."""
        self.output.setPlainText(translation)
        self.service = service
        self.btn_copy.setEnabled(True)
        # Прежний mp3 относится к старому тексту — «Повторить» пока недоступна.
        self.btn_repeat.setEnabled(False)
        if self.audio.available:
            self.set_status(f"Загружаю озвучку…  ·  перевод: {service}", "busy")
        else:
            self.set_status(f"Перевод: {service}", "ok")

    def on_audio_ready(self, path: str):
        """mp3 сгенерирован — делаем его текущим и сразу проигрываем."""
        self.audio.set_current(path)
        if self.audio.play(self.speed):
            self.set_status(f"Воспроизведение  ·  перевод: {self.service}", "ok")
            self.btn_repeat.setEnabled(True)
        else:
            self.set_status("Не удалось воспроизвести аудио", "error")

    def on_failed(self, message: str, translation_ok: bool):
        self.set_status(message.splitlines()[0], "error")
        QMessageBox.warning(self, "Ошибка", message)

    def on_worker_finished(self):
        self._set_busy(False)
        if self.status_kind == "busy":
            self.set_status("Готово", "ok")
        self.worker = None
        self.pending_mp3 = None

    def on_repeat(self):
        """Повторное воспроизведение уже готового mp3, без сети."""
        if self.audio.play(self.speed):
            self.set_status(f"Повтор озвучки  ·  темп {self.speed:g}×", "ok")
        else:
            self.set_status("Нет аудио для повтора", "error")
            self.btn_repeat.setEnabled(False)

    def on_speed_changed(self, speed: float):
        """Выбран другой темп: запоминаем и сразу даём его услышать."""
        self.speed = speed
        self.settings.setValue("speed", speed)
        if self.btn_repeat.isEnabled() and self.audio.play(speed):
            self.set_status(f"Темп озвучки {speed:g}×", "ok")
        else:
            self.set_status(f"Темп озвучки {speed:g}× — применится к следующему переводу", "idle")

    def on_copy(self):
        text = self.output.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.set_status("Перевод скопирован в буфер обмена", "ok")

    def _set_busy(self, busy: bool):
        self.btn_translate.setEnabled(not busy)
        self.btn_mic.setEnabled(not busy)
        self._set_direction_enabled(not busy)
        self.btn_translate.setText("Перевожу…" if busy else "Перевести")

    def _set_direction_enabled(self, enabled: bool):
        # Во время записи или перевода направление менять нельзя.
        self.btn_direction.setEnabled(enabled)
        self.btn_swap.setEnabled(enabled)

    # ----- закрытие окна -----

    def closeEvent(self, event):
        # Дожидаемся фонового потока, чтобы он не писал в уже удалённую папку.
        for thread in (self.worker, self.listener):
            if thread is not None and thread.isRunning():
                thread.wait(5000)
        self.audio.cleanup()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def install_excepthook():
    """
    PyQt6 аварийно завершает процесс при любом необработанном исключении
    внутри слота. Перехватываем такие ошибки и показываем их в окне,
    чтобы приложение не закрывалось внезапно.
    """
    import traceback

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Непредвиденная ошибка", text[-2000:])

    sys.excepthook = hook


def main():
    install_excepthook()
    app = QApplication(sys.argv)
    app.setApplicationName("Тиргум")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
