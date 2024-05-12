import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import jiwer
from vosk import Model, KaldiRecognizer
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, split="test-other", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device
        self.max_audio_length = max(len(audio[0]) for audio, _, _, _, _, _ in self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = self.preprocess_audio(audio)
        return audio, text

    def preprocess_audio(self, audio):
        audio = audio[0]  # Выбираем только первый канал
        if len(audio) < self.max_audio_length:
            audio = torch.nn.functional.pad(audio, (0, self.max_audio_length - len(audio)))
        return audio

def transcribe_audio(audio_data, model):
    recognizer = KaldiRecognizer(model, 16000)  # Указываем частоту дискретизации аудио
    recognizer.SetWords(True)  # Включаем вывод слов
    recognizer.AcceptWaveform(audio_data.numpy().tobytes())  # Передаем аудиоданные
    result = recognizer.Result()  # Получаем результат после обработки
    result = json.loads(result)
    transcription = result['text']
    return transcription

def remove_punctuation(test_str):
    result = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), test_str))
    return result.lower()

def calculate_wer(predicted_text, reference_text):
    transformation = jiwer.wer(reference_text, remove_punctuation(predicted_text))
    return transformation

def main():
    model = Model("vosk-model-small-en-us-0.15")

    dataset_clean = LibriSpeech("test-other")
    loader_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=16)

    hypotheses_vosk = []
    references_vosk = []

    for audios, texts in tqdm(loader_clean):
        for audio, text in zip(audios, texts):
            transcription = transcribe_audio(audio, model)
            hypotheses_vosk.append(transcription)
            references_vosk.append(text)

    wer_scores = []
    for hypothesis, reference in zip(hypotheses_vosk, references_vosk):
        if hypothesis and reference:
            wer_scores.append(calculate_wer(hypothesis, remove_punctuation(reference)))
        else:
            # Для пустых транскрипций добавляем максимальное значение WER (1.0)
            wer_scores.append(1.0)

    mean_wer = np.mean(wer_scores)
    print(f"Средний WER: {mean_wer * 100:.2f}%")

if __name__ == "__main__":
    main()