"""Metric for exact matches among documents"""
from typing import List


def doc_exact_match(y_pred_list: List, y_true_list: List) -> dict:
    """ 
    Gives the number of documents that have 100% accuracy. 
    We assume that the predictions and labels are given by word.
    i.e : ['Alex', 'is', 'in' 'New York'] is tokenized in 
          ['Alex', 'is', 'in', 'New', '##York'] the y_pred is
          ['B-PER', 'O', 'O', 'B-LOC'] 


    Parameters
    ----------
    y_pred_list : List of list
        The list of predictions from the model
    y_true_list: List of list
        The list of gold labels from the labeled dataset
    """
    assert len(y_pred_list) == len(y_true_list)
    for i in range(len(y_pred_list)):
        assert len(y_pred_list[i]) == len(y_true_list[i])

    y_tuples = [(y_pred_list[i], y_true_list[i])
                for i in range(len(y_true_list))]
    sum_metric = 0
    for i, (y_pred, y_true) in enumerate(y_tuples):

        if y_pred == y_true:
            sum_metric += 1
        # else:
        #     print(i)
        #     print(f'Pred : {y_pred}')
        #     print(f'True : {y_true}')
        #     print('\n')

    return sum([y_pred == y_true for (y_pred, y_true) in y_tuples])/len(y_tuples)
