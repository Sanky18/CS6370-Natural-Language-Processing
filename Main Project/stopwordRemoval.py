from util import *
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from math import log
from collections import Counter

class StopwordRemoval():
    def fromList(self, text):
        return [[word for word in sentence if
                 word.lower() not in stop_words] for sentence in text]

    def IDF_based(self, text):
        """
        Stopword Removal using IDF-based Strategy

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """
        # Flatten the list of sentences into a single list of words
        all_words = [word.lower() for sentence in text for word in sentence]
        # Calculate document frequency (DF) for each word
        df = Counter(all_words)
        # Total number of documents
        N = len(text)
        # Calculate Inverse Document Frequency (IDF) for each word
        idf = {word: log(N / (df[word] + 1)) for word in df}
        # Set a threshold for IDF values (you can experiment with this threshold)
        threshold = 2.0
        # Select words with IDF values above the threshold as stopwords
        stop_words = {word for word, score in idf.items() if score > threshold}
        # Remove stopwords from each sentence
        filtered_text = [
            [word for word in sentence if word.lower() not in stop_words]
            for sentence in text
        ]
        return filtered_text