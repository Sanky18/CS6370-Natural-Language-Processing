from util import *
from evaluation import Evaluation
import matplotlib.pyplot as plt
evaluator = Evaluation()
def Evaluation_metrics(doc_IDs_ordered, query_ids, qrels, op_folder='./', save_results=1, verbose=1, title_name=" "):
    """
    Evaluate retrieval metrics.

    Parameters:
    - doc_IDs_ordered (list): The order of the retrieved docs by our model.
    - query_ids (list): Values from 1 to 225. [1, 2, 3, ..., 225].
    - qrels (list): Relevant documents for each query (/cranfield/cran_qrels.json).
    - op_folder (str): Output folder path to save the results. Default is './'.
    - save_results (int): Determines whether to save plots along with table results.
        - 0: Output only the results table, not the plots.
        - 1: Plots + Table Results. Default is 1.
    - verbose (int): Verbosity level.
        - 0: Suppress verbose output.
        - 1: Print verbose output. Default is 1.
    - title_name (str): Title name of the plot (applicable when save_results = 1).

    Returns:
    None
    """
    # Initialize lists to store metrics
    precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []

    # Compute metrics for each value of k
    for k in range(1, 11):
        precision = evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
        precisions.append(precision)
        recall = evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
        recalls.append(recall)
        fscore = evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
        fscores.append(fscore)
        MAP = evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
        MAPs.append(MAP)
        nDCG = evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
        nDCGs.append(nDCG)
        # Print verbose output if specified
        if verbose:
            print(f"Precision, Recall, and F-score @ {k}: {precision}, {recall}, {fscore}")
            print(f"MAP, nDCG @ {k}: {MAP}, {nDCG}")

    # Plot the metrics and save plot if specified
    if save_results == 1:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title(title_name)
        plt.xlabel("k")
    return
