# Facteur Tabimetrique API

Une API REST complète pour les **Facteurs Tabimétriques**, une méthode avancée de sélection de variables combinant plusieurs mesures de corrélation.

## 📋 Table des matières

- [À propos](#à-propos)
- [Fondamentaux théoriques](#fondamentaux-théoriques)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Architecture](#architecture)
- [Endpoints](#endpoints)
- [Exemples](#exemples)
- [Tests](#tests)
- [Stack technique](#stack-technique)
- [Auteur](#auteur)

## À propos

Les **Facteurs Tabimétriques (FT)** mesurent la capacité explicative intrinsèque d'une variable en combinant :

- **ζ (zeta)** : Corrélation de Pearson (relation linéaire)
- **τ (tau)** : Corrélation de Kendall (relation monotone)
- **dCor** : Distance Correlation (relation globale)
- **C** : Dépendance transitive = |dCor - max(|τ|, |ζ|)|

## Fondamentaux théoriques

### Formule canonique

```
FT_j = tanh(w_j·τ_j + (1-w_j)·ζ_j + γ_j·C_j)
```

### Poids appris par MLP

Les coefficients w et γ sont appris par un réseau de neurones (MLP) à partir des **méta-caractéristiques** :

- S_lin = ζ² (degré de linéarité)
- S_norm = test de normalité (Shapiro-Wilk)
- S_out = sensibilité aux outliers (méthode IQR)

## Installation

### 1. Prérequis

- Python 3.8+
- pip ou conda

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Démarrage rapide

### Démarrer le serveur

```bash
python run.py
```

Le serveur démarre sur `http://localhost:8000`

### Accéder à la documentation

- **Swagger UI** : http://localhost:8000/api/docs
- **ReDoc** : http://localhost:8000/api/redoc

### Exécuter les exemples

```bash
python examples.py
```

### Lancer les tests

```bash
pytest tests.py -v
```

## Architecture

```
Facteur_Tabimetrique/
├── app/
│   ├── config.py                    # Configuration
│   ├── main.py                      # Application FastAPI
│   ├── core/
│   │   └── facteur_tabimetrique.py # Implémentation mathématique
│   ├── api/
│   │   └── routes.py               # Endpoints FastAPI
│   ├── models/
│   │   ├── requests.py             # Schémas de requête
│   │   └── responses.py            # Schémas de réponse
│   └── services/
│       ├── storage.py              # Gestion mémoire
│       └── ft_service.py           # Logique métier
├── requirements.txt
├── run.py
├── examples.py
├── tests.py
└── README.md
```

## Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/api/v1/train` | Entraîner un modèle FT |
| POST | `/api/v1/score` | Calculer les scores tabimétriques |
| POST | `/api/v1/select` | Sélectionner les variables (seuil) |
| POST | `/api/v1/pipeline` | Pipeline complet (train+score+select) |
| GET | `/api/v1/importance/{model_id}` | Tableau détaillé d'importance |
| POST | `/api/v1/compare` | Comparer FT vs Pearson/Spearman/DistCorr |
| POST | `/api/v1/upload-csv` | Upload CSV pour analyse |
| GET | `/api/v1/models` | Lister modèles en mémoire |
| DELETE | `/api/v1/models/{model_id}` | Supprimer un modèle |
| GET | `/health` | Health check |

## Exemples

### Python - Client simple

```python
from examples import FTAPIClient
import numpy as np

client = FTAPIClient()

# Générer données
X = np.random.randn(100, 5).tolist()
y = np.random.randn(100).tolist()

# Entraîner
response = client.train(model_id="my_model", X=X, y=y, epochs=50)
print(f"Status: {response['status']}")

# Scorer
scores = client.score(model_id="my_model", X=X)
print(f"FT Scores: {scores['ft_scores']}")

# Sélectionner
selected = client.select_features(model_id="my_model", threshold=0.5)
print(f"Selected: {selected['selected_features']}")
```

### cURL

```bash
# Entraîner
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_1",
    "X": [[1.0, 2.0], [2.0, 3.0]],
    "y": [1.0, 2.0],
    "epochs": 50
  }'

# Scorer
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_1",
    "X": [[1.0, 2.0], [2.0, 3.0]]
  }'

# Health check
curl http://localhost:8000/health
```

## Tests

```bash
# Tous les tests
pytest tests.py -v

# Tests spécifiques
pytest tests.py::TestFacteurTabimetrique -v

# Avec couverture
pytest tests.py --cov=app --cov-report=html
```

## Stack technique

| Composant | Version |
|-----------|---------|
| FastAPI | 0.104.1 |
| Pydantic | 2.5.0 |
| TensorFlow | 2.14.0 |
| NumPy | 1.24.3 |
| Pandas | 2.1.3 |
| SciPy | 1.11.4 |
| dcor | 0.6 |

## Configuration

Variables d'environnement dans `.env` :

```env
APP_NAME=Facteur Tabimetrique API
APP_VERSION=1.0.0
DEBUG=False
LOG_LEVEL=INFO
MODEL_STORAGE_LIMIT=50
MLP_EPOCHS=100
MLP_BATCH_SIZE=32
MLP_LEARNING_RATE=0.001
```

## Fonctionnalités

✅ Entraînement de modèles FT  
✅ Scoring adaptatif basé sur MLP  
✅ Sélection de variables automatique  
✅ Comparaison avec autres méthodes  
✅ Upload de fichiers CSV  
✅ Rapport détaillé d'importance  
✅ Gestion de stockage mémoire  
✅ Documentation OpenAPI  
✅ Suite complète de tests  
✅ Gestion d'erreurs complète  
✅ Logging structuré  
✅ CORS configuré  

## Auteur

**EYAGA TABI Jean François Régis**

- Email: francoistabi294@gmail.com
- GitHub: https://github.com/vulgane034
- LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235

## License

MIT License

---

**Dernière mise à jour:** 18 Janvier 2026
