import whisper


model = whisper.load_model("tiny")

result = model.transcribe("rep.wav")
podcast_transcript = result["text"]
podcast_transcript[:1000]
