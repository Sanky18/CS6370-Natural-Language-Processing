from util import *
from nltk.tokenize import PunktSentenceTokenizer
import re
import nltk
nltk.download('punkt')



class SentenceSegmentation():

    def naive(self, text):
        if isinstance(text, str):
            segments = re.split(delimiters, text)
            segmentedText = [s.strip() for s in segments]
            while '' in segmentedText:
                segmentedText.remove('')
        else:
            print("text is missing")
            segmentedText = []
        return segmentedText


    def punkt(self, text):
        if (isinstance(text, str)):
            tokenizer = PunktSentenceTokenizer(text)
            segmentedText = tokenizer.tokenize(text)
            return segmentedText
        else:
            print("text is missing")
            segmentedText = []
            return segmentedText