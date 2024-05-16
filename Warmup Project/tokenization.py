from util import *
from nltk.tokenize import TreebankWordTokenizer
import re

class Tokenization():
    
    def naive(self, text):
        tokenizedText = []
        if isinstance(text, list):
            for sentence in text:
                if isinstance(sentence, str):
                    segment_tokens = re.split(text_separators, sentence)
                    for token in segment_tokens:
                        if token in punctuations:
                            segment_tokens.remove(token)
                    tokenizedText.append(segment_tokens)
        else:
            print("text is missing")
        
        return tokenizedText 
    
    def pennTreeBank(self, text):
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

