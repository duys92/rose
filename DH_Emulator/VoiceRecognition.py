from deepspeech import Model as ds_Model
import numpy as np
import speech_recognition as sr
from text2digits import text2digits
import re


class VoiceRecognition(object):
    sample_rate = 16000
    beam_width = 500
    lm_alpha = 0.75
    lm_beta = 1.85
    n_features = 26
    n_context = 9
    audio = None
    r = sr.Recognizer()
    w2n = text2digits.Text2Digits()

    depth_control = None
    prev_depth_control = None

    def __init__(self, model_name, alphabet, lm, trie=None):
        print("Loading Model...this may take a while.")

        self.Model = ds_Model(model_name, self.n_features, self.n_context, alphabet, self.beam_width)
        self.Model.enableDecoderWithLM(alphabet, lm, trie, self.lm_alpha, self.lm_beta)

    def listen(self):
        text_num = ""
        try:
            with sr.Microphone(sample_rate=self.sample_rate) as source:
                print("*", end='')
                audio = self.r.listen(source, timeout=5)
                fs = audio.sample_rate
                assert fs == self.sample_rate, "Sample rate of {} in audio doesn't match {} of supplied".format(fs, self.sample_rate)
                self.audio = np.frombuffer(audio.frame_data, np.int16)
            text = self.Model.stt(self.audio, self.sample_rate)
            text_num = self.w2n.convert(text)

        except sr.WaitTimeoutError as err:
            pass
        return text_num

    def extract_numbers(self, text):
        text = text.split()
        numbers = []
        for i in text:
            try:
                numbers.append(float(i))
            except ValueError:
                pass

        return numbers

    def listen_and_do(self):
        while (True):
            text = self.listen()

            if not any(re.findall(r'rose', text, re.IGNORECASE)):
                continue

            if any(re.findall(r'end|quit|stop|sleep', text, re.IGNORECASE)):
                print("Goodbye")
                return

            possible_numbers = self.extract_numbers(text)

            self.depth_control = possible_numbers


            print("\n" + str(self.depth_control))


if __name__ == '__main__':
    vr = VoiceRecognition(model_name="lm_model/output_graph.pbmm",
                          alphabet="lm_model/alphabet.txt",
                          trie="lm_model/trie",
                          lm="lm_model/lm.binary")

    print(vr.listen())



