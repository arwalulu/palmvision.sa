from datetime import datetime, timedelta
import random
from pathlib import Path
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
import tensorflow as tf
from src.configs.loader import load_config
from src.models.cbam import SpatialAttention
from src.models.effb0_cbam import CLASS_NAMES  # ["Bug", "Dubas", "Healthy", "Honey"]
# MODEL / CONFIG FOR INFERENCE
_cfg = load_config()
IMG_SIZE = int(_cfg.training["img_size"])

MODEL = None  # lazy-loaded Keras model


def get_model():
    """
    Lazy-load the trained EffB0+CBAM model from:
        <project_root>/models/palmvision_effb0_cbam.keras
    """
    global MODEL
    if MODEL is None:
        model_path = (
            Path(__file__).resolve().parent.parent / "models" / "palmvision_effb0_cbam.keras"
        )
        if not model_path.exists():
            raise RuntimeError(f"Model file not found at: {model_path}")

        MODEL = tf.keras.models.load_model(
            model_path,
            custom_objects={"SpatialAttention": SpatialAttention},
            compile=False,  # inference only
        )

    return MODEL
# FASTAPI APP & CORS
app = FastAPI(title="PalmVision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# AUTH
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_name: str


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """
    Simple fake login for demo:
    - email: farmer@example.com
    - password: 1234
    """
    if req.email != "farmer@example.com" or req.password != "1234":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    return LoginResponse(
        access_token="fake-token-for-demo",
        user_name="Farmer Ali",
    )
# FARM SUMMARY SCHEMAS
class PerClassStats(BaseModel):
    bug: int
    dubas: int
    healthy: int
    honey: int


class FarmSummary(BaseModel):
    farm_name: str
    total_trees: int
    total_leaves_scanned: int
    healthy_leaves: int
    diseased_leaves: int
    per_class: PerClassStats

    overall_status: str
    risk_level: str
    last_scan_time: datetime
    next_recommended_scan: datetime

    soil_moisture: int
    avg_leaf_temp: float
    irrigation_status: str

    drone_status: str
    battery_level: int
    last_flight_duration_min: int

    report_url: str | None = None


class DroneStatus(BaseModel):
    drone_status: str
    battery_level: int
    last_flight_duration_min: int
    last_scan_time: datetime
# IN-MEMORY FARM STATE
_fake_state = {
    "farm_name": "Ali Date Farm",
    "total_trees": 320,
    "total_leaves_scanned": 124,
    "healthy_leaves": 98,
    "diseased_leaves": 26,
    "per_class": {
        "bug": 10,
        "dubas": 8,
        "healthy": 80,
        "honey": 26,
    },
    "overall_status": "Stable",
    "risk_level": "Medium",
    "last_scan_time": datetime.now() - timedelta(hours=3),
    "next_recommended_scan": datetime.now() + timedelta(days=2),
    "soil_moisture": 62,
    "avg_leaf_temp": 29.4,
    "irrigation_status": "Needs attention",
    "drone_status": "Offline",
    "battery_level": 73,
    "last_flight_duration_min": 18,
    "report_url": "/reports/latest",
}
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
# CHATBOT (RULE-BASED)
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Simple domain-aware agro assistant.
    Uses current farm state (_fake_state) instead of only hardcoded replies.
    """
    msg = req.message.lower().strip()
    s = _fake_state  # shorthand

    # ---------- 1) General farm health ----------
    if any(phrase in msg for phrase in ["overall status", "farm status", "how is my farm", "summary"]):
        diseased_ratio = s["diseased_leaves"] / max(s["total_leaves_scanned"], 1)
        reply = (
            f"Farm: {s['farm_name']}\n"
            f"- Overall status: {s['overall_status']} (risk: {s['risk_level']})\n"
            f"- Leaves scanned: {s['total_leaves_scanned']} (diseased: {s['diseased_leaves']}, "
            f"healthy: {s['healthy_leaves']})\n"
            f"- Next recommended scan: {s['next_recommended_scan'].strftime('%Y-%m-%d %H:%M')}\n"
            f"Estimated diseased ratio: {diseased_ratio:.1%}."
        )
        return ChatResponse(reply=reply)

    # ---------- 2) How many diseased / healthy / total ----------
    if "how many" in msg or "number of" in msg:
        if "diseased" in msg or "sick" in msg:
            reply = (
                f"There are currently {s['diseased_leaves']} diseased leaves "
                f"out of {s['total_leaves_scanned']} scanned."
            )
            return ChatResponse(reply=reply)

        if "healthy" in msg:
            reply = (
                f"There are currently {s['healthy_leaves']} healthy leaves "
                f"out of {s['total_leaves_scanned']} scanned."
            )
            return ChatResponse(reply=reply)

        if "leaves" in msg or "scanned" in msg:
            reply = (
                f"In total, {s['total_leaves_scanned']} leaves have been scanned so far "
                f"({s['healthy_leaves']} healthy, {s['diseased_leaves']} diseased)."
            )
            return ChatResponse(reply=reply)

    # ---------- 3) Risk level questions ----------
    if "risk" in msg or "danger" in msg:
        reply = (
            f"Current risk level is **{s['risk_level']}** with status **{s['overall_status']}**.\n"
            "If risk is Medium or High, you should inspect diseased palms and review irrigation and pest control."
        )
        return ChatResponse(reply=reply)

    # ---------- 4) Disease / classes questions ----------
    if any(w in msg for w in ["disease", "bug", "dubas", "honey", "class", "classes", "model detect"]):
        pc = s["per_class"]
        reply = (
            "The AI model is trained to classify four palm leaf classes:\n"
            f"- Bug: {pc['bug']} leaves detected so far\n"
            f"- Dubas: {pc['dubas']} leaves detected so far\n"
            f"- Honeydew / sooty mold: {pc['honey']} leaves detected so far\n"
            f"- Healthy: {pc['healthy']} leaves detected so far\n\n"
            "You can upload a leaf image in the dashboard to get a per-leaf prediction and recommendation."
        )
        return ChatResponse(reply=reply)

    # ---------- 5) Irrigation / soil moisture ----------
    if any(w in msg for w in ["irrigation", "water", "soil moisture", "moisture"]):
        reply = (
            f"Soil moisture is currently around {s['soil_moisture']}% and irrigation status is "
            f"'{s['irrigation_status']}'.\n"
            "If moisture is below 50–60%, you should irrigate the affected zones, especially around stressed palms."
        )
        return ChatResponse(reply=reply)

    # ---------- 6) Drone questions ----------
    if any(w in msg for w in ["drone", "battery", "flight", "scan"]):
        reply = (
            f"The drone is currently **{s['drone_status']}**.\n"
            f"Battery level: {s['battery_level']}%.\n"
            f"Last flight duration: {s['last_flight_duration_min']} minutes.\n"
            f"Last scan time: {s['last_scan_time'].strftime('%Y-%m-%d %H:%M')}."
        )
        return ChatResponse(reply=reply)

    # ---------- 7) Next scan / schedule ----------
    if "next scan" in msg or "when scan" in msg or "next recommended scan" in msg:
        reply = (
            f"The next recommended scan is on "
            f"{s['next_recommended_scan'].strftime('%Y-%m-%d %H:%M')}.\n"
            "Frequent scans help catch early disease spread, especially for Dubas and honeydew."
        )
        return ChatResponse(reply=reply)

    # ---------- 8) Default fallback ----------
    reply = (
        "I'm your PalmVision assistant. You can ask things like:\n"
        "- \"What is the overall farm status?\"\n"
        "- \"How many diseased leaves do I have?\"\n"
        "- \"What is the current risk level?\"\n"
        "- \"When should I irrigate?\"\n"
        "- \"What classes does the model detect?\"\n"
        "- \"What is the drone battery and last scan time?\""
    )
    return ChatResponse(reply=reply)


# FARM SUMMARY HELPERS & ENDPOINTS
def _make_summary() -> FarmSummary:
    return FarmSummary(
        farm_name=_fake_state["farm_name"],
        total_trees=_fake_state["total_trees"],
        total_leaves_scanned=_fake_state["total_leaves_scanned"],
        healthy_leaves=_fake_state["healthy_leaves"],
        diseased_leaves=_fake_state["diseased_leaves"],
        per_class=PerClassStats(**_fake_state["per_class"]),
        overall_status=_fake_state["overall_status"],
        risk_level=_fake_state["risk_level"],
        last_scan_time=_fake_state["last_scan_time"],
        next_recommended_scan=_fake_state["next_recommended_scan"],
        soil_moisture=_fake_state["soil_moisture"],
        avg_leaf_temp=_fake_state["avg_leaf_temp"],
        irrigation_status=_fake_state["irrigation_status"],
        drone_status=_fake_state["drone_status"],
        battery_level=_fake_state["battery_level"],
        last_flight_duration_min=_fake_state["last_flight_duration_min"],
        report_url=_fake_state["report_url"],
    )

@app.get("/dashboard/summary", response_model=FarmSummary)
def get_dashboard_summary():
    return _make_summary()


@app.get("/drone/status", response_model=DroneStatus)
def get_drone_status():
    return DroneStatus(
        drone_status=_fake_state["drone_status"],
        battery_level=_fake_state["battery_level"],
        last_flight_duration_min=_fake_state["last_flight_duration_min"],
        last_scan_time=_fake_state["last_scan_time"],
    )

@app.post("/drone/start_scan", response_model=FarmSummary)
def start_drone_scan():
    # 1) simulate new leaves
    new_leaves = random.randint(15, 35)
    _fake_state["total_leaves_scanned"] += new_leaves
    # 2) random new disease stats
    new_bug = random.randint(0, 5)
    new_dubas = random.randint(0, 5)
    new_honey = random.randint(0, 5)
    new_diseased = new_bug + new_dubas + new_honey
    new_healthy = max(new_leaves - new_diseased, 0)

    _fake_state["per_class"]["bug"] += new_bug
    _fake_state["per_class"]["dubas"] += new_dubas
    _fake_state["per_class"]["honey"] += new_honey
    _fake_state["per_class"]["healthy"] += new_healthy

    _fake_state["healthy_leaves"] += new_healthy
    _fake_state["diseased_leaves"] += new_diseased

    _fake_state["last_scan_time"] = datetime.now()
    _fake_state["next_recommended_scan"] = datetime.now() + timedelta(days=2)

    used_battery = random.randint(5, 15)
    _fake_state["battery_level"] = max(_fake_state["battery_level"] - used_battery, 10)
    _fake_state["last_flight_duration_min"] = random.randint(10, 25)

    ratio = _fake_state["diseased_leaves"] / max(_fake_state["total_leaves_scanned"], 1)
    if ratio < 0.1:
        _fake_state["risk_level"] = "Low"
        _fake_state["overall_status"] = "Stable"
    elif ratio < 0.25:
        _fake_state["risk_level"] = "Medium"
        _fake_state["overall_status"] = "Stable"
    else:
        _fake_state["risk_level"] = "High"
        _fake_state["overall_status"] = "Warning"

    return _make_summary()
# 7-DAY HISTORY (CHART)
@app.get("/dashboard/history")
def get_disease_history():
    return {
        "days": ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"],
        "total_leaves_scanned": [120, 135, 140, 150, 160, 170, 180],
        "diseased_leaves": [32, 28, 30, 27, 25, 24, 23],
    }
# PDF REPORT
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@app.get("/reports/latest", response_class=FileResponse)
def get_latest_report():
    pdf_path = REPORTS_DIR / "farm_report_demo.pdf"

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="palmvision_report.pdf",
    )
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float   # 0–1, we'll send 0.0–1.0
    severity: str
    recommendation: str

@app.post("/predict/image", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    """
    Real inference endpoint:
    - Accepts an uploaded palm leaf image.
    - Resizes & normalizes like training.
    - Runs EffB0+CBAM model.
    - Returns only: class, confidence, severity, recommendation.
    """
    contents = await file.read()
    # 1) Open & validate image
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    # 2) Resize + normalize to [0,1]
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    # 3) Predict using real model
    model = get_model()
    preds = model.predict(arr)
    probs = preds[0]  # shape (num_classes,)
    # 4) Map to labels using CLASS_NAMES from effb0_cbam.py
    if len(probs) != len(CLASS_NAMES):
        raise HTTPException(
            status_code=500,
            detail=f"Model output dim {len(probs)} does not match CLASS_NAMES {len(CLASS_NAMES)}",
        )

    best_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[best_idx]
    confidence = float(probs[best_idx])  # 0..1
    # 5) Severity + recommendation
    cls_lower = predicted_class.lower()
    if cls_lower == "healthy":
        severity = "Low"
        recommendation = (
            "Palms look healthy. Keep monitoring and follow your normal irrigation schedule."
        )
    elif cls_lower == "honey":
        severity = "Medium"
        recommendation = (
            "Monitor honeydew / sooty mold areas. Consider targeted treatment and pruning if it spreads."
        )
    else:  # Bug / Dubas / other diseases
        severity = "High"
        recommendation = (
            "Detected signs of pest/disease. Inspect affected palms, remove heavily infected leaves, "
            "and consult an agricultural expert for a treatment plan."
        )

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=round(confidence, 3),
        severity=severity,
        recommendation=recommendation,
    )
