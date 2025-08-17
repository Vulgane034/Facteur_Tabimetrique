# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 18:56:59 2025

@author: EYAGA TABI Francois
"""

# 🔬 Facteur Tabimétrique (TABI)

Implémentation en Python d’une **nouvelle méthode de sélection de variables** basée sur la **tabimétrie** (concept développé par **EYAGA TABI François**).
Cette approche combine des mesures statistiques robustes (Kendall Tau, corrélation linéaire, distance de corrélation) avec un modèle de pondération **neuronal** afin d’attribuer un **score tabimétrique** à chaque variable.

---

## 🚀 Fonctionnalités

* Calcul automatique des **scores TABI** pour chaque variable.
* Prend en compte :

  * 📊 corrélation non paramétrique (**Kendall Tau**)
  * 📈 corrélation linéaire (**Pearson**)
  * 🔗 dépendance non linéaire (**Distance correlation**)
  * ⚖️ robustesse aux distributions (test de Shapiro, outliers)
* Pondération dynamique par un **réseau de neurones**.
* Sélection des variables selon un **seuil TABI ajustable**.
* Visualisation possible des scores TABI via un **barplot matplotlib**.

---

## 📦 Installation

Cloner le dépôt et installer les dépendances :

```bash
git clone https://github.com/vulgane034/facteur-tabimetrique.git
cd facteur-tabimetrique
pip install -r requirements.txt
```

### requirements.txt

```
numpy
pandas
scikit-learn
tensorflow
dcor
matplotlib
```

---

## 📖 Exemple d’utilisation

```python
from sklearn.datasets import load_breast_cancer
from facteur_tabimetrique import FacteurTabimetrique

# Charger un dataset
data = load_breast_cancer()
X, y = data.data, data.target
features = data.feature_names

# Instancier
ft = FacteurTabimetrique()

# Entraîner le modèle de pondération
ft.train_weight_model(X, y, epochs=30)

# Calculer les scores
scores = ft.tabi_refine_vector(X, y)

# Sélectionner les variables
X_selected = ft.selectionner(X, features, seuil=0.30)

# Visualiser les scores TABI
ft.plot_scores(features)
```

---

## 📊 Exemple de sortie

```
✅ Modèle de pondération entraîné
✅ Scores TABI calculés
🎯 Features conservées avec TABI ≥ 0.30 :
 - mean radius : 0.4123
 - mean texture : 0.3891
 - mean perimeter : 0.4762
 ...
```

---

## 🧠 Références scientifiques

* Kendall M. (1938), *A New Measure of Rank Correlation*
* Székely G. J., Rizzo M. L. (2009), *Brownian distance covariance*
* EYAGA TABI François (2025), *Concept Tabimétrique pour la sélection de variables*

---

## ✨ Auteur

**EYAGA TABI François**
Élève ingénieur polytechnicien, Data Scientist & Chercheur en IA.

📧 Contact : [francoistabi294@gmail.com](mailto:francoistabi294@gmail.com)
[https://www.linkedin.com/in/francois-tabi-03a4b7235/](https://www.linkedin.com/in/francois-tabi-03a4b7235/)
🌍 Présentation prévue à PyCon & soumission scientifique.
