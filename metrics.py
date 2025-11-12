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

def confusion_matrix(true, pred, ignore=["<pad>", "<sos>", "<eos>"]):
    # Calcul de la matrice de confusion

    # À compléter
    matrix = dict()
    labels = set(true).union(set(pred))
    for label1 in labels:
        matrix[label1] = dict()
        for label2 in labels:
            matrix[label1][label2] = 0
    for t, p in zip(true, pred):
        if t in ignore:
            continue
        matrix[t][p] += 1
    return matrix
