import pandas as pd
from evaluation import Evaluation
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde


def plot(qrels, doc_IDs_ordered, queries, k, model_name=' ', bin_size=20):
    """
    Runs observations like distribution of recall, precision, ndcg, etc.

    Input Arguments:
    - qrels: Relevant documents for each query, imported from "./cranfield/cran_qrels.json".
    - doc_IDs_ordered: List of lists, output is the list of retrieved documents for each query.
    - queries: List, containing all the queries.
    - k: int, number of top retrieved documents to be considered, usually ranges from 1 to 10.
    - model_name: str, name of the model being evaluated.
    - bin_size: int, number of bins for histogram plot.
    """
    df = pd.DataFrame(qrels)
    evaluator = Evaluation()

    ones_precision, zeros_precision = [], []
    ones_recall, zeros_recall = [], []
    q_precision, q_recall, q_fscore = [], [], []
    q_ndcg = []

    # Compute metrics for each query
    for i in range(len(doc_IDs_ordered)):
        true_doc_ids = list(map(int, df[df['query_num'] == str(i+1)]['id'].tolist()))
        precision = evaluator.query_precision(doc_IDs_ordered[i], i+1, true_doc_ids, k)
        q_precision.append(precision)
        recall = evaluator.query_recall(doc_IDs_ordered[i], i+1, true_doc_ids, k)
        q_recall.append(recall)
        fscore = evaluator.query_fscore(doc_IDs_ordered[i], i+1, true_doc_ids, k)
        q_fscore.append(fscore)

        if precision == 1:
            ones_precision.append({'q_id': i+1, 'query': queries[i],
                                    'rel_docs': true_doc_ids, 'ret_docs': doc_IDs_ordered[i][:10]})
        if precision == 0:
            zeros_precision.append({'q_id': i+1, 'query': queries[i],
                                    'rel_docs': true_doc_ids, 'ret_docs': doc_IDs_ordered[i][:10]})
        if recall == 1:
            ones_recall.append({'q_id': i+1, 'query': queries[i],
                                'rel_docs': true_doc_ids, 'ret_docs': doc_IDs_ordered[i][:10]})
        if recall == 0:
            zeros_recall.append({'q_id': i+1, 'query': queries[i],
                                 'rel_docs': true_doc_ids, 'ret_docs': doc_IDs_ordered[i][:10]})

        true_doc_ndcg = df[df['query_num'] == str(i+1)][['position', 'id']]
        ndcg = evaluator.query_nDCG(doc_IDs_ordered[i], i+1, qrels, k)
        q_ndcg.append(ndcg)

    # Plot distributions
    plot_distribution(q_precision, 'Precision', model_name)
    plot_distribution(q_recall, 'Recall', model_name)
    plot_distribution(q_fscore, 'Fscore', model_name)
    plot_distribution(q_ndcg, 'nDCG', model_name)

    return q_precision, q_recall, q_fscore, q_ndcg


def plot_comp(q_precision_1, q_precision_2, q_recall_1, q_recall_2, q_fscore_1, q_fscore_2, q_ndcg_1, q_ndcg_2, k, model1_name=' ', model2_name=' '):
    """
    Compare distribution of metrics between two models.

    Input Arguments:
    - q_precision_1: List, Precision scores for Model 1.
    - q_precision_2: List, Precision scores for Model 2.
    - q_recall_1: List, Recall scores for Model 1.
    - q_recall_2: List, Recall scores for Model 2.
    - q_fscore_1: List, Fscore scores for Model 1.
    - q_fscore_2: List, Fscore scores for Model 2.
    - q_ndcg_1: List, nDCG scores for Model 1.
    - q_ndcg_2: List, nDCG scores for Model 2.
    - k: int, number of top retrieved documents considered.
    - model1_name: str, name of Model 1.
    - model2_name: str, name of Model 2.
    """
    # Plot distribution comparison
    plot_comparison(q_precision_1, q_precision_2, 'Precision', model1_name, model2_name)
    plot_comparison(q_recall_1, q_recall_2, 'Recall', model1_name, model2_name)
    plot_comparison(q_fscore_1, q_fscore_2, 'Fscore', model1_name, model2_name)
    plot_comparison(q_ndcg_1, q_ndcg_2, 'nDCG', model1_name, model2_name)

    return

def plot_distribution(data, metric, model_name):
    """
    Plot distribution of a metric.

    Input Arguments:
    - data: List, metric scores.
    - metric: str, name of the metric.
    - model_name: str, name of the model.
    """
    data = np.array(data) 
    plt.figure(figsize=(10, 5))
    plt.xlabel(f'{metric} @ k')
    plt.ylabel('Frequency of Queries')
    plt.title(f'{metric} Distribution with KDE for {model_name} on Cranfield Dataset')
    
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)
    plt.plot(x_vals, kde(x_vals), color='darkblue', linewidth=3, label='KDE Curve')
    
    plt.hist(data, bins=20, density=True, alpha=0.3, color='lightblue', edgecolor='black', linewidth=1)
    plt.legend()
    plt.show()

def plot_comparison(data1, data2, metric, model1_name, model2_name):
    """
    Plot comparison of metrics between two models.

    Input Arguments:
    - data1: List, metric scores for Model 1.
    - data2: List, metric scores for Model 2.
    - metric: str, name of the metric.
    - model1_name: str, name of Model 1.
    - model2_name: str, name of Model 2.
    """
    plt.figure(figsize=(10, 5))
    plt.xlabel(f'{metric} @ k')
    plt.ylabel('Frequency of Queries')
    plt.title(f'Distribution Comparison of {metric} Scores: {model1_name} vs {model2_name} (Cranfield Dataset)')

    # Plot KDE for Model 1
    kde1 = gaussian_kde(data1)
    x_vals1 = np.linspace(min(data1), max(data1), 1000)
    plt.plot(x_vals1, kde1(x_vals1), color='red', label=f'{model1_name} KDE')
    # Plot KDE for Model 2
    kde2 = gaussian_kde(data2)
    x_vals2 = np.linspace(min(data2), max(data2), 1000)
    plt.plot(x_vals2, kde2(x_vals2), color='darkblue', label=f'{model2_name} KDE')
    # Plot histograms with lighter colors
    plt.hist(data1, bins=20, density=True, alpha=0.3, color='lightcoral', edgecolor='black', linewidth=3, label=model1_name)
    plt.hist(data2, bins=20, density=True, alpha=0.3, color='lightblue', edgecolor='black', linewidth=3, label=model2_name)
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()

