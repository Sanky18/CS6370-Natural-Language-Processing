from util import *
from nltk.stem import PorterStemmer


class InflectionReduction:
    def reduce(self, text):
        if isinstance(text, list):
            for i in range(len(text)):
                if isinstance(i, str):
                    for j in range(len(i)):
                        ps = PorterStemmer()
                        text[i][j] = ps.stem(text[i][j])  
            reducedText = text
        else:
            print("text is missing")
            reducedText = None 
        return reducedText

        




