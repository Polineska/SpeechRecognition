import os
import warnings
import jiwer
import numpy as np
import pandas as pd
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import jiwer
import io
import soundfile as sf

# Импорт библиотеки datasets
from datasets import load_dataset

# Импорт библиотеки speech_recognition
import speech_recognition as sr

#DATASET_ID = "bond005/sberdevices_golos_10h_crowd"
DATASET_ID = "bond005/sberdevices_golos_10h_crowd_noised_2db"
SAMPLES = 7000

test_dataset = load_dataset(DATASET_ID, split=f"test[:{SAMPLES}]")

# Функция для преобразования аудиофайлов в массивы
def speech_file_to_array_fn(batch):
    speech_array = batch["audio"]["array"]
    batch["speech"] = speech_array
    return batch

# Предобработка датасета
removed_columns = set(test_dataset.column_names)
removed_columns -= {'transcription', 'speech'}
removed_columns = sorted(list(removed_columns))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    test_dataset = test_dataset.map(
        speech_file_to_array_fn,
        remove_columns=removed_columns
    )

# Функция для распознавания речи с помощью массива numpy
def transcribe_audio(audio_data, language='ru-RU'):
    recognizer = sr.Recognizer()

    # Преобразование массива numpy в формат WAV
    with io.BytesIO() as wav_file:
        sf.write(wav_file, audio_data, 16000, format='WAV')
        wav_file.seek(0)

        with sr.AudioFile(wav_file) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio, language=language)
            return text
        except sr.UnknownValueError:
            print("Речь не распознана")
            return ""
        except sr.RequestError as e:
            print(f"Ошибка запроса к API распознавания речи; {e}")
            return ""

hypotheses = []
for sample in test_dataset:
    audio = np.array(sample["speech"], dtype=np.float32)  # Преобразование в numpy array
    transcript = transcribe_audio(audio)
    hypotheses.append(transcript)

references = test_dataset["transcription"]


references = test_dataset["transcription"]

import re

def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    # Удаление знаков пунктуации и преобразование в нижний регистр
    text = re.sub(r'[^\w\s]', '', text).lower()
    
    # Удаление множественных пробелов
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
normalized_refs = [normalize_text(ref) for ref in test_dataset["transcription"]]
normalized_hyps = [normalize_text(hyp) for hyp in hypotheses]

# Вычисление WER с нормализованными транскрипциями
wer_score = jiwer.wer(normalized_refs, normalized_hyps)
print(f"\nWord Error Rate: {wer_score * 100:.2f}%")
