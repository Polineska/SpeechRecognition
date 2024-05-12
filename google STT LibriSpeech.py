import os
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import speech_recognition as sr
from jiwer import wer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        return audio.numpy(), text

def transcribe_audio(audio_data, language='en'):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data, language=language)
        return text
    except sr.UnknownValueError:
        print("Речь не распознана")
    except sr.RequestError as e:
        print(f"Ошибка запроса к API распознавания речи; {e}")

def remove_punctuation(test_str):
    result = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), test_str))
    return result.lower()

def calculate_wer(predicted_text, reference_text):
    transformation = jiwer.wer(reference_text, remove_punctuation(predicted_text))
    return transformation

dataset_clean = LibriSpeech("test-clean")
loader_clean = torch.utils.data.DataLoader(dataset_clean, batch_size=16)

hypotheses_google = []
references_google = []

for audios, texts in tqdm(loader_clean):
    for audio, text in zip(audios, texts):
        transcription = transcribe_audio(audio)
        if transcription:
            hypotheses_google.append(transcription)
            references_google.append(text)

wer_scores = []
for hypothesis, reference in zip(hypotheses_google, references_google):
    wer_scores.append(calculate_wer(hypothesis, remove_punctuation(reference)))

mean_wer = np.mean(wer_scores)
print(f"Средний WER : {mean_wer * 100:.2f} %")
