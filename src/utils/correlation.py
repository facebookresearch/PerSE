from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np

def calculate_correlation(pred_score, human_score):
    result = {'pearsonr': 0, 'spearmanr': 0, 'kendalltau': 0}
    assert len(pred_score) == len(human_score)
    print(len(pred_score))

    pearsonr_res = pearsonr(pred_score, human_score)
    spearmanr_res = spearmanr(pred_score, human_score)
    kendalltau_res = kendalltau(pred_score, human_score)
    result['pearsonr'], result['pearsonr_status'] = pearsonr_res[0], pearsonr_res[1:]
    result['spearmanr'], result['spearmanr_status'] = spearmanr_res[0], spearmanr_res[1:]
    result['kendalltau'], result['kendalltau_status'] = kendalltau_res[0], kendalltau_res[1:]

    match = (pred_score == human_score).sum()
    accu = match / len(pred_score)
    result['accu'] = accu

    return result

def compute_metric(expected, labels):
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    return {
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


if __name__ == "__main__":
    pred_score = [0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,1]
    human_score = [1,1,1,1,0,0,0,1,1,0,1,0,1,1,0,1,1,1,0,1,1]
    result = calculate_correlation(pred_score, human_score)
    print(result)