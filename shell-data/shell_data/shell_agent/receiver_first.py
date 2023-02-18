"""
Three strategies to rank samples. Scorer assigns a score
to each data point (train + val) so the receiver can request such images from the sender 

- Least confident: prioritize the samples with the lowest confidence (i.e. lowest maximum logits)
- Margin sampling: prioritize the samples with the lowest margin (i.e. difference between the two highest logits)
- Entropy: prioritize the samples with the highest entropy (i.e. highest entropy of the softmax distribution)


Higher score means more valuable to train more on.



https://github.com/gtoubassi/active-learning-mnist/blob/master/Active_Learning_with_MNIST.ipynb
"""
import torch
import torch.nn.functional as F


def least_confidence_scorer(X, y, model):
    with torch.no_grad():
        logits = model(X)
        p = F.softmax(logits, dim=1)
        return -torch.max(p, dim=1).values


def margin_scorer(X, y, model):
    with torch.no_grad():
        logits = model(X)
        max_logits, _ = torch.max(logits, dim=1)
        second_max_logits = torch.topk(logits, k=2, dim=1).values[:, 1]
        return -(max_logits - second_max_logits)


def entropy_scorer(X, y, model):
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        return -torch.sum(probs * torch.log(probs), dim=1)


def wrong_prediction_scorer(X, y, model):
    # give 1 to wrong prediction, 0 to correct prediction
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        return (preds != y).float()


def combine_wrong_least_confidence_scorer(X, y, model):
    """
    1. Give higher score to wrong prediction than correct prediction
    2. Give higher score to wrong prediction with higher confidence to wrong prediction with lower confidence
    3. Give higher score to correct prediction with lower confidence to correct prediction with higher confidence

    Make sure that wrong prediction always have positive score and correct prediction always have negative score
    """
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        p = F.softmax(logits, dim=1)
        correct = (preds == y).float()
        wrong = (preds != y).float()
        confidence = torch.max(p, dim=1).values
        return wrong * confidence + correct * -confidence
