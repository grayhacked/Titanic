from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import logging
from pydantic import BaseModel, Field



# Initialiser FastAPI
app = FastAPI()

#configurations de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Charger le modèle entraîné avec gestion des erreurs
try:
    with open("Model/Tmodele.pkl", "rb") as file:
        modele = pickle.load(file)
    logging.info("Modèle chargé avec succès")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle: {e}")
    raise RuntimeError("Impossible de charger le modèle")

#Definition du shemade validation des entrées
class InputData(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex : str
    Age: int = Field(ge=0, le=100)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: str
    
#verifier que les valeurs categoriques sont valides
valid_sex = {"male" : 1, "female" : 0}
valid_embarked = {"S": 0, "C": 1, "Q": 2}

# Définir l'endpoint de prédiction
@app.post("/predict")
def predict(data: InputData):
    #verifier les valeurs des variables catégoriques
    if data.Sex not in valid_sex or data.Embarked not in valid_embarked:
        raise HTTPException(status_code=400, detail="Valeur invalide pour 'Sex' ou 'Embarked'")
    
    # Transformer les entrées en DataFrame
    df = pd.DataFrame([[data.Pclass, valid_sex[data.Sex], data.Age, data.SibSp, data.Parch, data.Fare, valid_embarked[data.Embarked]]], columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
    
    # Prédiction
    prediction = modele.predict(df)[0]
    
    #logger la prediction
    logging.info(f"Prediction: {prediction} pour {data}")
    
    # Retourner la réponse
    return {"Survivra": bool(prediction)}
