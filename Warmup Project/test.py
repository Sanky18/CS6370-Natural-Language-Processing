delimiters = '[.?!]'
from nltk.tokenize import PunktSentenceTokenizer
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
from util import *

def naive(text):
    if isinstance(text, str):
        segments = re.split(delimiters, text)
        segmentedText = [s.strip() for s in segments]
        while '' in segmentedText:
            segmentedText.remove('')
    else:
        print("text is missing")
        segmentedText = []
    return segmentedText

def punkt(text):
    if (isinstance(text, str)):
        tokenizer = PunktSentenceTokenizer(text)
        segmentedText = tokenizer.tokenize(text)
        print(tokenizer)
        return segmentedText 
    else:
        print("No text received")
        return ([])
    
def reduce(text):
    if isinstance(text, list):
        for i in range(len(text)):
            if isinstance(text[i], list):  # Check if text[i] is a list
                for j in range(len(text[i])):
                    ps = PorterStemmer()
                    text[i][j] = ps.stem(text[i][j])
        reducedText = text
    else:
        print("text is missing")
        reducedText = None  # Assign a default value if text is missing
        
    return reducedText


def pennTreeBank(text):
    tokenizedText = []
    if isinstance(text, list):
        for sentence in text:
            if isinstance(sentence, str):
                tokenizer = TreebankWordTokenizer()
                segment_tokens = tokenizer.tokenize(sentence)
            tokenizedText.append(segment_tokens)  
    else:
        print("text is missing")
                
    return tokenizedText

output = pennTreeBank(["The well-known actor starred in a record-breaking award-winning performance"])
print(output)