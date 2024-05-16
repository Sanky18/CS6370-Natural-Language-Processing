from util import *
from math import log2

import pandas as pd


class Evaluation():
    

	def query_precision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""
        # Compute the total number of documents retrieved
		num_docs = len(query_doc_IDs_ordered)
		# Check if the value of k is valid
		if k > num_docs:
			print("Insufficient documents retrieved for given k")
			return -1
		# Initialize a variable to count the number of relevant documents
		num_relevant_docs = 0
		# Iterate over the top k ranked documents
		for doc_id in query_doc_IDs_ordered[:k]:
			# Check if the document ID is among the relevant documents
			if int(doc_id) in true_doc_IDs:
				# If the document is relevant, increment the count
				num_relevant_docs += 1
		# Compute and return the precision value
		precision = num_relevant_docs / k
		return precision


	def meanPrecision(self, doc_ids_ordered, query_ids, qrels, k):
		"""
		Compute mean precision of the Information Retrieval System
		at a given value of k, averaged over all the queries.

		Parameters
		----------
		doc_ids_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query.
		query_ids : list
			A list of IDs of the queries for which the documents are ordered.
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary.
		k : int
			The k value.

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1.
		"""
		num_queries = len(query_ids)
		precisions = []
		# Check if the number of queries and documents match
		if len(doc_ids_ordered) != num_queries:
			print("Number of queries and documents do not match")
			return -1
		# Iterate over all queries
		for i in range(num_queries):
			query_docs = doc_ids_ordered[i]
			query_id = int(query_ids[i])
			# Retrieve relevant documents for the query
			relevant_docs = [int(entry["id"]) for entry in qrels if int(entry["query_num"]) == query_id]
			# Compute precision for the query
			precision = self.query_precision(query_docs, query_id, relevant_docs, k)
			precisions.append(precision)
		# Compute and return the mean precision value
		if precisions:
			return sum(precisions) / num_queries
		else:
			print("Empty list.")
			return -1


	def query_recall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Compute recall of the Information Retrieval System
		at a given value of k for a single query.

		Parameters
		----------
		ranked_doc_ids : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query.
		query_id : int
			The ID of the query in question.
		relevant_doc_ids : list
			The list of IDs of documents relevant to the query (ground truth).
		k : int
			The k value.

		Returns
		-------
		float
			The recall value as a number between 0 and 1.
		"""
		num_docs = len(query_doc_IDs_ordered)
		num_relevant_docs = len(true_doc_IDs)
		# Check if the number of documents retrieved is sufficient for k
		if k > num_docs:
			print("Insufficient number of retrieved documents for given k")
			return -1
		# Count the number of relevant documents retrieved within top k
		num_retrieved_relevant_docs = 0
        # Iterate over the top k ranked documents
		for doc_id in query_doc_IDs_ordered[:k]:
            # Check if the document ID is among the relevant documents
			if int(doc_id) in true_doc_IDs:
                # If the document is relevant, increment the count
				num_retrieved_relevant_docs += 1
		# Compute and return recall value
		return num_retrieved_relevant_docs / num_relevant_docs if num_relevant_docs != 0 else 0
    
    
	def meanRecall(self, doc_ids_ordered, query_ids, qrels, k):
		"""
		Compute mean recall of the Information Retrieval System
		at a given value of k, averaged over all the queries.

		Parameters
		----------
		doc_ids_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query.
		query_ids : list
			A list of IDs of the queries for which the documents are ordered.
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary.
		k : int
			The k value.

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1.
		"""
		num_queries = len(query_ids)
		recalls = []
		# Check if the number of queries and documents match
		if len(doc_ids_ordered) != num_queries:
			print("Number of queries and documents do not match")
			return -1
		# Iterate over all queries
		for i in range(num_queries):
			query_docs = doc_ids_ordered[i]
			query_id = query_ids[i]
			# Retrieve relevant documents for the query
			relevant_docs = [int(entry["id"]) for entry in qrels if int(entry["query_num"]) == query_id]
			# Compute recall for the query
			recall = self.query_recall(query_docs, query_id, relevant_docs, k)
			recalls.append(recall)
		# Compute and return the mean recall value
		if recalls:
			return sum(recalls) / num_queries
		else:
			print("List is empty.")
			return -1


	def query_fscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""
		# Compute precision and recall for the query
		precision = self.query_precision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.query_recall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		# Compute fscore based on precision and recall
		if precision > 0 and recall > 0:
			fscore = 2 * precision * recall / (precision + recall)
		else:
			fscore = 0
		return fscore


	def meanFscore(self, doc_ids_ordered, query_ids, qrels, k):
		"""
		Compute fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries.

		Parameters
		----------
		doc_ids_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query.
		query_ids : list
			A list of IDs of the queries for which the documents are ordered.
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary.
		k : int
			The k value.

		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1.
		"""
		num_queries = len(query_ids)
		fscores = []
		# Check if the number of queries and documents match
		if len(doc_ids_ordered) != num_queries:
			print("Number of queries and documents do not match")
			return -1
		# Iterate over all queries
		for i in range(num_queries):
			query_docs = doc_ids_ordered[i]
			query_id = query_ids[i]
			# Retrieve relevant documents for the query
			relevant_docs = [int(entry["id"]) for entry in qrels if int(entry["query_num"]) == query_id]
			# Compute fscore for the query
			fscore = self.query_fscore(query_docs, query_id, relevant_docs, k)
			fscores.append(fscore)
		# Compute and return the mean fscore value
		if fscores:
			return sum(fscores) / num_queries
		else:
			print("Empty list")
			return -1
	

	def query_nDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Compute nDCG (Normalized Discounted Cumulative Gain) for a single query.

		Parameters
		----------
		query_doc_IDs_ordered : list
			A list of integers denoting the IDs of documents in their predicted order of relevance to a query.
		query_id : int
			The ID of the query in question.
		true_doc_IDs : list
			A list of dictionaries containing ground truth relevance information.
		k : int
			The k value.

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1.
		"""
		# Create a dictionary to store relevant documents for the given query
		relevant_docs = {}
		for doc in true_doc_IDs:
			if int(doc["query_num"]) == int(query_id):
				doc_id = int(doc["id"])
				relevance = 5 - doc["position"]  # Calculate relevance score
				relevant_docs[doc_id] = relevance
		# Calculate DCG (Discounted Cumulative Gain) at position k
		DCG_k = 0
		for rank in range(1, min(k, len(query_doc_IDs_ordered)) + 1):
			doc_ID = int(query_doc_IDs_ordered[rank - 1])
			if doc_ID in relevant_docs:
				relevance = relevant_docs[doc_ID]
				DCG_k += (2 ** relevance - 1) / log2(rank + 1)  # Update DCG_k
		# Calculate IDCG (Ideal Discounted Cumulative Gain) at position k
		sorted_relevances = sorted(relevant_docs.values(), reverse=True)
		IDCG_k = sum((2 ** relevance - 1) / log2(rank + 1) for rank, relevance in enumerate(sorted_relevances, 1) if rank <= k)
		# Check if IDCG_k is not zero, then compute nDCG
		if IDCG_k != 0:
			nDCG_k = DCG_k / IDCG_k
			return nDCG_k
		else:
			# If IDCG_k is zero, print a message and return -1
			print("IDCG_k is zero.")
			return -1




	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_IDs_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgments - Refer cran_qrels.json for the structure of each dictionary
		k : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""
		num_queries = len(query_ids)
		ndcgs = []
		# Check if the number of queries and documents match
		if len(doc_IDs_ordered) != num_queries:
			print("Number of queries and documents do not match")
			return -1
		# Iterate over all queries
		for i in range(num_queries):
			query_doc = doc_IDs_ordered[i]
			query_id = int(query_ids[i])
			nDCG = self.query_nDCG(query_doc, query_id, qrels, k)
			ndcgs.append(nDCG)
		# Compute and return the mean nDCG value
		if ndcgs:
			return sum(ndcgs) / num_queries
		else:
			print("Empty list")
			return -1



	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		query_doc_IDs_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_IDs : list
			The list of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""
		num_true_docs = len(true_doc_IDs)
		num_docs_retrieved = len(query_doc_IDs_ordered)
		# Check if enough documents are retrieved for the given k
		if k > num_docs_retrieved:
			print("Insufficient documents retrieved for given k")
			return -1
		relevances = [1 if int(doc_ID) in true_doc_IDs else 0 for doc_ID in query_doc_IDs_ordered]
		# Calculate precision@i for each document up to k
		precisions = [self.query_precision(query_doc_IDs_ordered, query_id, true_doc_IDs, i) for i in range(1, k + 1)]
		# Calculate precision at k multiplied by relevance for each document
		precision_at_k = [precisions[i] * relevances[i] for i in range(k)]
		# Calculate average precision
		if num_true_docs != 0:
			if sum(relevances[:k]) != 0:
				AveP = sum(precision_at_k) / len(true_doc_IDs)
			else:
				AveP = 0
			return AveP
		else:
			print("No true documents are present.")
			return -1



	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""
		num_queries = len(query_ids)
		map = []
		# Check if the number of queries and documents match
		if len(doc_IDs_ordered) != num_queries:
			print("Number of queries and documents do not match")
			return -1
		# Iterate over all queries
		for i in range(num_queries):
			query_docs = doc_IDs_ordered[i]
			query_id = query_ids[i]
			# Retrieve relevant documents for the query
			relevant_docs = [int(entry["id"]) for entry in q_rels if int(entry["query_num"]) == query_id]
			# Compute average precision for the query
			ap = self.queryAveragePrecision(query_docs, query_id, relevant_docs, k)
			map.append(ap)
		# Compute and return the mean average precision value
		if map:
			return sum(map) / num_queries
		else:
			print("Empty list")
			return -1

