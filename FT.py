# -*- coding: utf-8 -*-
"""
Implémentation de la classe FacteurTabimetrique
Auteur : EYAGA TABI Francois
Description : Classe pour le calcul et la sélection de variables selon la méthode tabimétrique.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from scipy.stats import kendalltau, darling
from dcor import distance_correlation

class FacteurTabimetrique:
    def __init__(self):
        self.model = self._build_weight_model()
        self.features = None
        self.tabi_scores = None
        self.mlp_weights = None

    def _build_weight_model(self):
        """Construit le modèle de pondération (réseau de neurones)."""
        input_stats = Input(shape=(3,))
        x = Dense(32, activation='relu')(input_stats)
        x = Dense(16, activation='relu')(x)
        w_output = Dense(1, activation='sigmoid')(x)
        gamma_output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=input_stats, outputs=[w_output, gamma_output])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_weight_model(self, X, y, epochs=50):
        """Entraîne le modèle de pondération sur les méta-caractéristiques."""
        features_list, tau_list, zeta_list, dcor_list = [], [], [], []

        y = np.ravel(y)

        for j in range(X.shape[1]):
            xj = np.ravel(X[:, j])

            tau, _ = kendalltau(xj, y)
            zeta = np.corrcoef(xj, y)[0, 1]
            dcor_val = distance_correlation(xj, y)

            # Méta-caractéristiques
            S_lin = zeta**2
            p_x = darling(xj)
            S_norm = p_x
            S_out = np.mean(np.abs(xj - np.mean(xj)) > 2 * np.std(xj))

            features_list.append([S_lin, S_norm, S_out])
            tau_list.append(tau)
            zeta_list.append(zeta)
            dcor_list.append(dcor_val)

        meta_X = np.array(features_list)
        tau_array = np.array(tau_list)
        zeta_array = np.array(zeta_list)
        dcor_array = np.array(dcor_list)

        # Cibles d’entraînement
        target_weight = tau_array / (tau_array + zeta_array + 1e-8)
        target_gamma = np.abs(dcor_array - np.maximum(np.abs(tau_array), np.abs(zeta_array)))

        self.model.fit(meta_X, [target_weight, target_gamma], epochs=epochs, verbose=0)
        print(" Modèle de pondération entraîné")
        return self.model

    def tabi_refine_vector(self, X, y):
        """Calcule les scores tabimétriques pour chaque feature."""
        scores = []
        y = np.ravel(y)

        for j in range(X.shape[1]):
            xj = np.ravel(X[:, j])
            tau, _ = kendalltau(xj, y)
            zeta = np.corrcoef(xj, y)[0, 1]
            dcor_val = distance_correlation(xj, y)
            C = abs(dcor_val - max(abs(tau), abs(zeta)))

            # Méta-caractéristiques
            S_lin = zeta**2
            p_x = darling(xj)
            S_norm = p_x
            S_out = np.mean(np.abs(xj - np.mean(xj)) > 2 * np.std(xj))
            meta_features = np.array([[S_lin, S_norm, S_out]])

            w, gamma = self.model.predict(meta_features, verbose=0)
            w = w[0][0]
            gamma = gamma[0][0]

            score = w * tau + (1 - w) * zeta + gamma * C
            scores.append(score)

        self.tabi_scores = np.array(scores)
        print(" Scores TABI calculés")
        return self.tabi_scores

    def selectionner(self, X, feature_names, seuil):
        """Sélectionne les features dont le score TABI dépasse un seuil."""
        if self.tabi_scores is None:
            raise ValueError(" Calculer d’abord les scores TABI avec `tabi_refine_vector`")

        mask = np.abs(self.tabi_scores) >= seuil
        self.features = feature_names

        X_df = pd.DataFrame(X, columns=feature_names)
        X_selected = X_df.loc[:, mask]

        print(f" Features conservées avec TABI ≥ {seuil} :")
        for name, score, keep in zip(feature_names, self.tabi_scores, mask):
            if keep:
                print(f" - {name} : {score:.4f}")

        return X_selected
    def _get_mlp_weights(self):
        if self.model is None:
            return None
        
        weights_dict={}
        for layer in self.model.layers:
            if len(layer.get_weights())> 0:
                weights_dict[layer.name]={
                    'weights':layer.get_weights()[0].tolist(),
                    'biais':layer.get_weights()[1].tolist() if len(layer.get_weights())>1 else None
                }
        self.mlp_weights=weights_dict
        return weights_dict
    
