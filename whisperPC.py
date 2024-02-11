import whisper


import io
import os

import torch
import pandas as pd
import urllib
import tarfile


model = whisper.load_model("tiny")

result = model.transcribe("rep.wav")
podcast_transcript = result["text"]
podcast_transcript[:1000]

from scipy.io import wavfile
from tqdm import tqdm

class Fleurs(torch.utils.data.Dataset):
	pass

pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000

# Set inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set language (korean)
language_google = "ru"
language_whisper = "ru"
dataset = Fleurs(language_google, subsample_rate=1, device=device)
dataset = torch.utils.data.random_split(dataset, [10, len(dataset)-10])[0]

# Set options
options = dict(language="ru", beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)
translate_options = dict(task="translate", **options)

references = []
transcriptions = []
translations = []

for audio, text in tqdm(dataset):
    transcription = model.transcribe(audio, **transcribe_options)["text"]
    translation = model.transcribe(audio, **translate_options)["text"]

    transcriptions.append(transcription)
    translations.append(translation)
    references.append(text)