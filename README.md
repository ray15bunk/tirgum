# Тиргум — голосовой переводчик RU ⇄ EN с озвучкой

Говорите по-русски — слушайте по-английски (и наоборот).

**▶ Открыть в браузере: https://ray15bunk.github.io/tirgum/**

**⬇ Скачать для Windows: [Tirgum.exe](https://github.com/ray15bunk/tirgum/releases/latest/download/Tirgum.exe)**

## Возможности

- 🎤 Голосовой ввод: нажмите микрофон (или F2), скажите фразу, и после паузы она переведётся и прозвучит.
- ⌨ Текстовый ввод: Enter переводит, Shift+Enter добавляет новую строку.
- ⇄ Направление перевода RU → EN или EN → RU (кнопка в шапке или F4).
- 🔊 Автоматическая озвучка перевода и повтор без нового запроса.
- Темп озвучки 1× / 0.85× / 0.7×. По умолчанию 0.85×, чуть медленнее, при этом голос не искажается.
- 📋 Копирование перевода в один клик.

## Две версии

| | Веб-версия (`docs/index.html`) | Программа для Windows (`tirgum.py` / `Tirgum.exe`) |
|---|---|---|
| Запуск | по ссылке, ничего не устанавливается | скачать `Tirgum.exe` и запустить |
| Голосовой ввод | Chrome и Edge (Web Speech API) | любой микрофон (SpeechRecognition + PyAudio) |
| Перевод | Google Translate, резерв MyMemory | Google Translate, резерв deep-translator (Google, MyMemory) |
| Озвучка | голоса браузера | gTTS + pygame, замедление без изменения высоты (WSOLA) |

Обеим версиям нужен интернет.

## Запуск из исходников (Windows)

```
python -m pip install -r requirements.txt
python tirgum.py
```

## Сборка .exe

```
python -m pip install pyinstaller
python -m PyInstaller --onefile --windowed --name Tirgum --collect-data speech_recognition tirgum.py
```

Готовый файл появится в `dist\Tirgum.exe`.

## Если Windows показывает «Windows защитила ваш компьютер»

Программа не подписана цифровой подписью, поэтому SmartScreen её не знает. Нажмите «Подробнее», затем «Выполнить в любом случае».
