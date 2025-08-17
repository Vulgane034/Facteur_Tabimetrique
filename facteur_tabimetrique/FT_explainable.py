# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:01:39 2025

@author: EYAGA TABI Francois
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

class TABI:
    """
    Implémentation du Facteur Tabimétrique (TABI).
    """
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self.scores_ = None

    def fit(self, X: np.ndarray):
        """
        Calcule les coefficients TABI à partir des données X.

        Paramètres
        ----------
        X : np.ndarray
            Données d'entrée (n_samples, n_features).
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Calcul de la matrice de distances pairwise
        dist_matrix = pairwise_distances(X, metric=self.metric)

        # Exemple simplifié de calcul : moyenne normalisée
        self.scores_ = dist_matrix.mean(axis=1) / dist_matrix.max()
        return self

    def transform(self) -> np.ndarray:
        """
        Retourne les scores TABI calculés.
        """
        if self.scores_ is None:
            raise ValueError("Le modèle TABI doit être ajusté avant d'utiliser transform().")
        return self.scores_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform()

    def plot_scores(self, labels=None):
        """
        Visualisation des scores TABI.

        Paramètres
        ----------
        labels : list, optional
            Étiquettes pour chaque point.
        """
        if self.scores_ is None:
            raise ValueError("Aucun score disponible. Utilisez fit() ou fit_transform() avant.")

        plt.figure(figsize=(8, 5))
        plt.bar(range(len(self.scores_)), self.scores_)
        if labels is not None:
            plt.xticks(range(len(labels)), labels, rotation=45)
        plt.title("Scores Tabimétriques (TABI)")
        plt.xlabel("Échantillons")
        plt.ylabel("Score TABI normalisé")
        plt.tight_layout()
        plt.show()


