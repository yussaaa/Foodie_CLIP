from torcheval.metrics import MultilabelAccuracy
import torch
import numpy as np

def compute_accuracy(eval_preds):
    logits, labels = eval_preds
    # get the predicted caption for each image
    pred = logits.argmax(axis=-1, keepdims=True)
    result = np.take_along_axis(labels, indices=pred, axis=-1)
    n_image = pred.shape[0]
    accuracy = result.sum(axis=0)/n_image
    return accuracy

if __name__ == '__main__':  
    s = np.array([[3,2,1],[3,6,3],[1,2,4]])
    t = np.array([[1,0,0],[0,0,1],[0,0,1]])
    print(compute_accuracy((s, t)))

