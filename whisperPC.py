import whisper

# Загрузка аудиофайла с использованием модели "tiny"

model = whisper.load_model("small")

result = model.transcribe("rep.wav")
podcast_transcript = result["text"]
podcast_transcript[:1000]
print(podcast_transcript)
