from os import path
import jiwer
from pydub import AudioSegment
import speech_recognition as sr
from jiwer import wer


def transcribe_audio(audio_file_path, language='en'):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as audio_file:
        audio_data = recognizer.record(audio_file)
        try:
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            print("Речь не распознана")
        except sr.RequestError as e:
            print(f"Ошибка запроса к API распознавания речи; {e}")


#sound = AudioSegment.from_mp3("аудиоданные/links_en.mp3") #конвектирование файла в wav формат
#sound.export("аудиоданные/links_en.wav", format="wav")

audio_file_path = "аудиоданные/termin_en.wav"

transcription = transcribe_audio(audio_file_path)
print(f"Распознанный текст: {transcription}")

def remove_punctuation(test_str): #удаление пунктуации, приведение теста к нижнему регистру
# Using filter() and lambda function to filter out punctuation characters
  result = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), test_str))
  return result.lower()

def calculate_wer(predicted_text, reference_file): #функция для расчета wer
    with open(reference_file, 'r', encoding='utf-8') as file:
        reference_text = remove_punctuation(file.read())
    transformation = jiwer.wer(reference_text, remove_punctuation(predicted_text))
    return transformation

reference_file='аудиоданные/termin_en.txt'
wer_score = calculate_wer(transcription, reference_file)
print(f"WER : {wer_score*100:.2f} %")






