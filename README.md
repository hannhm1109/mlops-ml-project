# mlops-ml-project (baseline)

Projet ML baseline avec workflow Git/MLOps.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Entraînement
```bash
python scripts/train.py
```

## Évaluation
```bash
python scripts/evaluate.py
```

## Artefacts générés

- `artifacts/model.joblib` - Modèle entraîné (pipeline complet)
- `artifacts/metrics.json` - Métriques principales (accuracy, F1)
- `artifacts/confusion_matrix.png` - Visualisation matrice de confusion
- `artifacts/report.json` - Rapport de classification détaillé

## Dataset

Iris (sklearn) - 150 échantillons, 3 classes, 4 features.