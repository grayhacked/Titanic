from pathlib import Path
import logging
import pickle

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Model" / "Tmodele.pkl"

SEX_MAP = {"male": 1, "female": 0}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}
MODEL_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("titanic-api")


class PredictionInput(BaseModel):
    Pclass: int = Field(ge=1, le=3)
    Sex: str
    Age: int = Field(ge=0, le=120)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: str

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in SEX_MAP:
            raise ValueError("Sex must be one of: male, female")
        return cleaned

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if cleaned not in EMBARKED_MAP:
            raise ValueError("Embarked must be one of: S, C, Q")
        return cleaned


class PredictionOutput(BaseModel):
    survived: bool
    prediction_label: str
    survival_probability: float | None = None


app = FastAPI(
    title="Titanic Survival API",
    description="Predict if a passenger survived using a trained model",
    version="1.1.0",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def load_model(model_path: Path):
    if not model_path.exists():
        raise RuntimeError(f"Model not found at: {model_path}")

    try:
        return joblib.load(model_path)
    except Exception:
        with model_path.open("rb") as model_file:
            return pickle.load(model_file)


try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception as error:
    logger.exception("Unable to load model")
    model = None


def prepare_features(payload: PredictionInput) -> pd.DataFrame:
    row = {
        "Pclass": payload.Pclass,
        "Sex": SEX_MAP[payload.Sex],
        "Age": payload.Age,
        "SibSp": payload.SibSp,
        "Parch": payload.Parch,
        "Fare": payload.Fare,
        "Embarked": EMBARKED_MAP[payload.Embarked],
    }
    return pd.DataFrame([row], columns=MODEL_COLUMNS)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Check startup logs and model file.")

    try:
        features = prepare_features(payload)
        raw_prediction = int(model.predict(features)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features)[0][1])

        result = PredictionOutput(
            survived=bool(raw_prediction),
            prediction_label="Survived" if raw_prediction == 1 else "Did not survive",
            survival_probability=probability,
        )

        logger.info("Prediction completed: input=%s output=%s", payload.model_dump(), result.model_dump())
        return result
    except Exception as error:
        logger.exception("Prediction failure")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error
