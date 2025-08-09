from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
import io
import soundfile as sf

from src.models.model_manager import ModelManager
from src.features.feature_extractor import FeatureExtractor

app = FastAPI(title="Speech Emotion Classification API")

model_manager = ModelManager()
loaded = False
model = None
feature_extractor = FeatureExtractor()
emotion_labels = None

class PredictResponse(BaseModel):
    emotion: str
    probabilities: dict

@app.on_event("startup")
def load_model_on_startup():
    global model, loaded, emotion_labels
    latest = model_manager.get_latest_model(model_type="cnn") or model_manager.get_latest_model()
    if latest:
        model = model_manager.load_model(model_id=latest['id'])
        feat_info = model_manager.load_feature_info(model_path=latest['path'])
        if feat_info and 'normalization_params' in feat_info:
            feature_extractor.set_normalization_params(feat_info['normalization_params'])
        loaded = model is not None
    # Lazy import to avoid heavy deps before load
    from src.core import config as core_config
    emotion_labels = core_config.Config().training.emotion_labels

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not loaded or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        data = await file.read()
        audio, sr = sf.read(io.BytesIO(data))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        features = feature_extractor.extract_features(audio, sr)
        features = feature_extractor.normalize_single(features, feature_type='mel_spectrogram')
        preds = model.predict(features, verbose=0)[0]
        labels = emotion_labels[: len(preds)]
        emotion = labels[int(np.argmax(preds))]
        probs = {labels[i]: float(preds[i]) for i in range(len(labels))}
        return PredictResponse(emotion=emotion, probabilities=probs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8501, reload=False)


