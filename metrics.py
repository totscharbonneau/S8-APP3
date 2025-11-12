# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import numpy as np

def edit_distance(a,b):
    # Calcul de la distance d'édition

    len_a = len(a) + 1
    len_b = len(b) + 1
    dp = np.zeros((len_a, len_b), dtype=int)
    for i in range(len_a):
        dp[i][0] = i
    for j in range(len_b):
        dp[0][j] = j

    for i in range(1, len_a):
        for j in range(1, len_b):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len_a - 1][len_b - 1]

def confusion_matrix(true, pred, ignore=[0, 1, 2], num_classes=None): # Ignore padding and start/end tokens
    # Calcul de la matrice de confusion

    all_true = []
    all_pred = []
    for i in range(len(true)):
        for j in range(min(len(true[i]), len(pred[i]))):
            if true[i][j] not in ignore:  # only ignore as input
                all_true.append(true[i][j])
                all_pred.append(pred[i][j])

    # automatically infer class count if not provided
    if num_classes is None:
        num_classes = max(max(all_true, default=0), max(all_pred, default=0)) + 1

    # initialize full confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # fill matrix
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    classes = list(range(num_classes))
    cm = cm[3:, 3:] 
    classes = classes[3:]
    return cm, classes