import pandas as pd
import numpy as np
import json
import string
import random
import re
import time
import itertools
from Levenshtein import edit_distance
from collections import Counter


class BigramSpellCheck:
    def __init__(self, documents):
        # Create the vocabulary from the documents
        words = []
        for document in documents:
            # Clean the document by removing non-alphabetic characters
            cleaned_words = re.sub("[^a-z ]+", " ", document).split()
            words.extend(cleaned_words)
        
        # Build the vocabulary as a set of unique words
        self.vocabulary = list(set(words))
        print("Vocabulory Constructed...")
        # Generate all possible bigrams from aa to zz
        self.bigrams = ["".join(bigram_tuple) for bigram_tuple in itertools.product(string.ascii_lowercase, repeat=2)]

        # Create the bigram reverse index
        self.bigram_reverse_index = {}
        for bigram in self.bigrams:
            # Find words in the vocabulary that contain the current bigram
            self.bigram_reverse_index[bigram] = [word for word in self.vocabulary if bigram in word]
         

    def candidate_correction(self, query_word):
        """
        Returns the top 5 candidate corrections
        """
        # If the query word is in the vocabulary with correct spelling
        if query_word in self.vocabulary:
            print("Word doesn't have any typos, so no candidates.")
            return []

        # Get all the bigrams for the input query word
        bigrams_of_query = [query_word[i:i+2] for i in range(len(query_word)-1)]

        # Get all possible candidates for the query word
        candidates = []
        for query_bigram in bigrams_of_query:
            candidates.extend(self.bigram_reverse_index.get(query_bigram, []))
        candidates = list(set(candidates))

        # Calculate and store similarity scores for each candidate
        candidates_with_scores = []
        for candidate in candidates:
            bigrams_of_candidate = [candidate[i:i+2] for i in range(len(candidate)-1)]
            common_bigrams = [bigram for bigram in bigrams_of_query if bigram in bigrams_of_candidate]
            similarity_score = len(common_bigrams) / len(bigrams_of_query)
            candidates_with_scores.append((candidate, similarity_score))

        # Sort candidates by similarity score in descending order
        sorted_candidates = sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)

        # Return the top 5 candidates with the highest similarity scores
        top_candidates = [candidate[0] for candidate in sorted_candidates[:5]]
        return top_candidates
    
        
    def edit_distance_candiate(self, query_word):
        """
        Returns the top candidate among 5 candidates using edit distance formula
        """
        # If the query word is in the vocabulary with correct spelling
        if query_word in self.vocabulary:
            return query_word  # No correction needed for correctly spelled word
        # Calculate edit distances between the query word and candidate corrections
        edit_distances = [(edit_distance(query_word, candidate), candidate) 
                          for candidate in self.candidate_correction(query_word)]
        # Sort the candidates based on edit distances in ascending order
        edit_distances.sort()
        # Return the corrected word with the smallest edit distance (top candidate)
        return edit_distances[0][1]

# Load the Cranfield dataset documents
docs_json = json.load(open("cranfield/cran_docs.json", 'r'))[:]
doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]
# Given typos
typos = ['boundery', 'transiant', 'aerplain']
# Create an instance of BigramSpellCheck with the Cranfield dataset documents
spell_checker = BigramSpellCheck(docs)
# Find and print the top 5 candidate corrections for each typo
for typo in typos:
    candidates = spell_checker.candidate_correction(typo)
    print(f"Top 5 candidate corrections for '{typo}': {candidates}")
    # Find and print the closest candidate using Edit Distance
    closest_candidate = spell_checker.edit_distance_candiate(typo)
    print(f"Closest candidate to '{typo}' using Edit Distance: {closest_candidate}")
    print("="*50)