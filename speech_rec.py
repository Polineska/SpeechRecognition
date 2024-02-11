#import speech_recognition as sr 
#import moviepy.editor as mp
#clip = mp.VideoFileClip(r"Доклад.mov") 
#clip.audio.write_audiofile(r"converted.wav")
import speech_recognition as sr

def transcribe_audio(audio_file_path, language='ru-RU'):
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


audio_file_path = "doclad.wav"

transcription = transcribe_audio(audio_file_path)

print(f"Распознанный текст: {transcription}")


def calculate_wer(reference, hypothesis):
	ref_words = reference.split() #слова  из исходного текста
	hyp_words = hypothesis.split()#слова из распозноваемого текста
	# Counting the number of substitutions, deletions, and insertions
  #
	S = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp) #подсчет кол-во замен слов 
	D = len(ref_words) - len(hyp_words) #количество удалений
	I = len(hyp_words) - len(ref_words) #количество вставок 
	words = len(ref_words) # всего слов в тексте исходном
	WER = (S + D + I) / words
	return WER
reference_text = "В настоящее время каждый житель нашей планеты производит в среднем, около 1 тонны бытовых отходов в год. Бытовые отходы — это многокомпонентная смесь, содержащая полимерные материалы, металлы, стекло, бумагу и пищевые отходы. В бытовых отходах содержится большое количество опасных химических веществ : ртуть из батареек, люминофоры из флуоресцентных ламп, токсичные бытовые растворители, химические компоненты органорастворимых красок и  материалы для защиты деревянных покрытий, которые влияют на здоровье населения и загрязняют окружающую среду." #исходный текст
hypothesis_text = transcription 


wer = calculate_wer(reference_text, hypothesis_text)
print("Word Error Rate (WER):", wer)

#расчет wer с помощью расстояния Левенштейна
import numpy as np
def calculate_wer(reference, hypothesis):
    # Split the reference and hypothesis sentences into words
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # Initialize a matrix with size |ref_words|+1 x |hyp_words|+1
    # The extra row and column are for the case when one of the strings is empty
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    # The number of operations for an empty hypothesis to become the reference
    # is just the number of words in the reference (i.e., deleting all words)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    # The number of operations for an empty reference to become the hypothesis
    # is just the number of words in the hypothesis (i.e., inserting all words)
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    # Iterate over the words in the reference and hypothesis
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            # If the current words are the same, no operation is needed
            # So we just take the previous minimum number of operations
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                # If the words are different, we consider three operations:
                # substitution, insertion, and deletion
                # And we take the minimum of these three possibilities
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    # The minimum number of operations to transform the hypothesis into the reference
    # is in the bottom-right cell of the matrix
    # We divide this by the number of words in the reference to get the WER
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer
if __name__ == "__main__":
  reference = "В настоящее время каждый житель нашей планеты производит в среднем, около 1 тонны бытовых отходов в год. Бытовые отходы — это многокомпонентная смесь, содержащая полимерные материалы, металлы, стекло, бумагу и пищевые отходы. В бытовых отходах содержится большое количество опасных химических веществ : ртуть из батареек, люминофоры из флуоресцентных ламп, токсичные бытовые растворители, химические компоненты органорастворимых красок и  материалы для защиты деревянных покрытий, которые влияют на здоровье населения и загрязняют окружающую среду." 
  hypothesis_text = transcription 
 

  wer1 = calculate_wer(reference, hypothesis_text)
  print("Word Error Rate (WER):", wer1)



