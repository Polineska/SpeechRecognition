# SpeechRecognition

## Содержание

- [Собственный набор данных](#собственный-набор-данных)
- [Google Speech-to-Text](#google-speech-to-text)
- [Whisper](#whisper)
- [VOSK](#vosk)

### Собственный набор данных
own dataset – датасет из отобранных аудиоданных

### Google Speech-to-Text

- `google_golos.py`: aнализ Google STT на русскоязычном датасете GOLOS
- `google STT LibriSpeech.py`: aнализ Google STT на англоязычном датасете LibriSpeech
- `Google_Speach_to_text.py`: kокальная установка Google STT и анализ на выборочных аудиоданных

### Whisper

- `whisperPC.py`: Локальная установка Whisper
- `Whisper_for_app.ipynb`: Код, предназначенный для внедрения в приложение
- `whisper_datasetRUGolos.ipynb`: анализ whisper на русскоязычном датасете GOLOS
- `Whisper_analysis.ipynb`: анализ whisper на англоязычном датасете LibriSpeech, анализ на выборочных аудиоданных, на собственном датасете, визуализация WER

### VOSK

- `Vosk.ipynb`: локальная установка VOSK и анализ на выборочных аудиоданных
- `vosk LibriSpeech.py`: анализ VOSK на англоязычном датасете LibriSpeech
- `vosk_analysisGolos.ipynb`: анализ VOSK на русскоязычном датасете GOLOS

Этот репозиторий предоставляет инструменты и ресурсы для распознавания речи, а также анализы производительности различных инструментов на различных наборах данных, включая собственный набор данных. Он может послужить отправной точкой для дальнейшей работы в области распознавания речи или быть полезным для сравнения и выбора подходящего инструмента для ваших задач
