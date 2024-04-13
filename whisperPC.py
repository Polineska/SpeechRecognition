#Работа с Whisper локально. Предварительно необходимо установить соответсвтующие библеотеки на ПК, через консольную строку 
import whisper
model = whisper.load_model("small") #загрузка модели whisper

result = model.transcribe("rep.wav") #файл, который необходимо распознать. Важно файл должен находиться в одной папке с кодом
podcast_transcript = result["text"] #сохранение расшифрованного текста
podcast_transcript[:1000]
print(podcast_transcript)
