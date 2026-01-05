# started 1/4/2026

from fastapi import FastAPI
from pydantic import BaseModel
from mvp_service import MVPService
from fastapi.middleware.cors import CORSMiddleware



import sys
app = FastAPI()           # create the web app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for local testing)
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)
print("[SERVER] Initializing MVPService...", file=sys.stderr)
service = MVPService()    # load your model, scalers, and explainer once
print("[SERVER] MVPService initialized.", file=sys.stderr)


class Features(BaseModel):
    temp_max_F: float
    humidity_pct: float
    windspeed_mph: float
    precip_in: float
    ndvi: float
    pop_density: float
    slope: float

@app.post("/predict")
def predict(features: Features):
    print("[SERVER] /predict called", file=sys.stderr)
    print("[SERVER] Features:", features.dict(), file=sys.stderr)
    result = service.predict(features.dict())
    print("[SERVER] Prediction result:", result, file=sys.stderr)
    return result

@app.post("/explain")
def explain(features: Features):
    print("[SERVER] /explain called", file=sys.stderr)
    features_dict = features.dict()
    # Log and coerce all values to float
    for k, v in features_dict.items():
        print(f"[SERVER] Feature {k}: {v} ({type(v)})", file=sys.stderr)
        try:
            features_dict[k] = float(v)
        except Exception as e:
            print(f"[SERVER] Could not convert {k} to float: {e}", file=sys.stderr)
    print("[SERVER] Features (as float):", features_dict, file=sys.stderr)
    result = service.explain(features_dict)
    print("[SERVER] Explain result:", result, file=sys.stderr)
    return result