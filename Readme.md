# Titanic Model Deployment

Ce projet expose un modele de prediction de survie Titanic via une API FastAPI et une interface web interactive.

## Structure utile

- `api.py` : API principale (prediction, health check, UI)
- `templates/index.html` : interface interactive
- `static/style.css` : style de l'interface
- `Model/Tmodele.pkl` : modele entraine
- `Model/Titanic.ipynb` : notebook d'exploration/entrainement

## Demarrage rapide

1. Activer l'environnement virtuel

```powershell
& "./venv/Scripts/Activate.ps1"
```

1. Installer les dependances (si besoin)

```powershell
pip install -r requirements.txt
```

1. Lancer le serveur

```powershell
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

1. Ouvrir les pages

- Interface interactive : <http://127.0.0.1:8000/>
- Documentation Swagger : <http://127.0.0.1:8000/docs>
- Health check : <http://127.0.0.1:8000/health>

## Contrat API

### POST `/predict`

Payload JSON attendu:

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 36,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

Exemple de reponse:

```json
{
  "survived": false,
  "prediction_label": "Did not survive",
  "survival_probability": 0.22
}
```

## Ameliorations implementees

1. Robustesse du chargement modele

- Resolution du chemin via `pathlib` pour eviter les erreurs de chemin relatif.
- Message d'erreur explicite si le fichier `.pkl` est absent.

1. Validation metier stricte

- Validation des champs numeriques (`Field`) et categoriels (`field_validator`).
- Nettoyage automatique des entrees (`Sex`, `Embarked`).

1. API plus professionnelle

- Versionnement et metadata FastAPI (`title`, `description`, `version`).
- `response_model` pour garantir un format de sortie stable.
- Endpoint `GET /health` pour supervision rapide.

1. Interface interactive integree

- Formulaire HTML pour tester le modele sans outil externe.
- Requete `fetch` vers l'endpoint `/predict`.
- Affichage du label et de la probabilite.

## Pistes d'amelioration suivantes

1. Pipeline de preprocessing reutilisable

- Deplacer l'encodage dans un pipeline scikit-learn persiste pour garantir la coherence train/inference.

1. Evaluation plus rigoureuse

- Ajouter validation croisee, AUC, precision/recall selon seuil metier.

1. Industrialisation

- Ajouter tests unitaires (`pytest`) pour validation des schemas et endpoint `/predict`.
- Ajouter `requirements.txt` epingle (`==`) et un `Dockerfile`.

1. Monitoring

- Journaliser latence, volume de requetes et taux d'erreur.
- Ajouter un endpoint de metriques (Prometheus) si deploiement en production.

## Note compatibilite modele

Le fichier `Model/Tmodele.pkl` a ete serialise avec scikit-learn 1.6.1.
Utiliser une version tres differente peut produire des warnings ou des comportements instables.
