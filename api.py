from fastapi import FastAPI
import pickle
import pandas as pd

# Charger le modèle entraîné
with open("Model/Tmodele.pkl", "rb") as file:
    modele = pickle.load(file)

# Vérifier le type de l'objet
print(type(modele))

# Initialiser FastAPI
app = FastAPI()

# Définir l'endpoint de prédiction
@app.post("/predict")
def predict(Pclass: int, Sex: str, Age: int, SibSp: int, Parch: int, Fare: float, Embarked: str):
    
    # Transformer les entrées en DataFrame
    data = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]], columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
    
    # Encodage des variables catégoriques (ex: Sex -> 0 ou 1)
    data["Sex"] = data["Sex"].map({"male": 1, "female": 0})
    #Encodage des variables catégoriques (ex: Embarked -> 0, 1 ou 2)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Prédiction
    prediction = modele.predict(data)[0]
    
    # Retourner la réponse
    return {"Survivra": bool(prediction)}
