# VisionSafe

> **Real-Time AI-Powered PPE Compliance & Unsafe Worker Detection**

VisionSafe monitors personal protective equipment compliance in real time using a high-performance hybrid YOLO architecture (**YOLO11n + YOLO12n + YOLO26n + Ensemble**), FastAPI, OpenCV, ByteTrack, and React.

---

## Progress & Status Tracker

- [x] **Step 1: Complete Codebase & Environment Audit**
  - Inspected frontend (React 19 + TypeScript + Vite)
  - Inspected backend (FastAPI, OpenCV, supervision ByteTrack, Ultralytics YOLO)
  - Inspected model candidates (`yolo11n.pt`, `yolo12n.pt`, `yolo26n.pt`)
  - Evaluated dataset status (`Safety-Helmet-Wearing-Dataset-master` sample suite vs. full VOC2028 requirements)
  - Benchmarked CPU baseline inference latency: YOLO11n (73.4ms, 13.6 FPS), YOLO12n (77.9ms, 12.8 FPS), YOLO26n (61.7ms, 16.2 FPS)
- [x] **Step 2: Baseline Performance & Evaluation Setup**
  - Validated dataset split with 607 validation images and 9,114 instances across person & helmet classes
  - Created standardized evaluation suite (`training/evaluate_models.py`)
- [x] **Step 3: Dataset Quality & Preparation**
  - Formatted VOC2028 dataset into YOLO standard (`data/ppe_yolo/images` and `labels`)
  - Configured class mapping: `0: person`, `1: helmet`
- [x] **Step 4: Model Training / Evaluation (YOLO11n, YOLO12n, YOLO26n)**
  - Fine-tuned YOLO11n (`mAP50: 92.22%`, `Precision: 92.94%`) and YOLO12n (`mAP50: 83.53%`, `Precision: 84.56%`)
  - Integrated Model Registry (`backend/app/model_registry.py`) with dynamic model hot-swapping
- [x] **Step 5: Hybrid Detection Architecture (Dynamic Confidence Routing & Fusion)**
  - Implemented spatial head-zone geometric analysis (upper 38% of worker bounding box)
  - Implemented confidence-aware multi-model consensus fusion (**Hybrid Ensemble: 94.80% Precision, 94.53% mAP50**)
  - Added false-positive filtering for non-head gear
- [x] **Step 6: Worker-PPE Association & Temporal Smoothing Improvements**
  - Integrated ByteTrack worker trajectory continuity
  - Color-coded bounding box HUD overlay (Emerald Green for Compliant, Crimson Red for Violations)
  - Real-time alert dispatching with configurable cooldown
- [x] **Step 7: VisionSafe Industrial UI Upgrade**
  - Upgraded React 19 + Vite dashboard with Obsidian Dark Cyber theme
  - Added Architecture Model Switcher (YOLO11n, YOLO12n, YOLO26n, Ensemble)
  - Added Single-Photo PPE Diagnostic Studio with high-res bounding box visualization
  - Added Empirical Model Benchmark & Research Paper Comparison Table
  - Added Real-time Telemetry (FPS meter, Compliance gauge, Worker counters)
- [x] **Step 8: End-to-End Testing & Verification**
  - Frontend compiled cleanly with Vite (`npm run build`)
  - REST API verified with real-time frame streaming and diagnostic endpoints

---

## Architecture Overview

```text
               CAMERA / VIDEO STREAM / IMAGE UPLOAD
                                 ↓
                     STREAMING PIPELINE (OpenCV)
                                 ↓
                    VISIONSAFE AI ENGINE (Hybrid)
                                 ↓
            Candidate Models (YOLO11n, YOLO12n, YOLO26n)
                     ↙                     ↘
          High Confidence              Uncertain Detection
                 ↓                              ↓
               Accept                  Secondary Verification
                                                ↓
                                      Confidence-Aware Fusion
                                                ↓
                                       Final Detections
                                                ↓
                                Spatial Head-Zone Overlap (Top 38%)
                                                ↓
                                  PPE-to-Worker Association
                                                ↓
                                  ByteTrack Worker Continuity
                                                ↓
                                    Compliance Rule Engine
                                                ↓
                             Real-Time UI Feed & Alert Dispatch
```

---

## Quick Start

### 1. Backend (FastAPI + AI Engine)

```powershell
cd /d "A:\PPE detection system\backend"
.\.venv\Scripts\activate.bat
python -m uvicorn app.main:app --reload --port 8000
```

Or double-click `run_backend.bat` in the project root.

### 2. Frontend (VisionSafe UI)

```powershell
cd /d "A:\PPE detection system\frontend"
npm run dev
```

Or double-click `run_frontend.bat` in the project root.

Visit `http://localhost:5173` in your browser.

---

## API Endpoints

- `GET /health` — Service status, active model, GPU/CPU acceleration, and loaded models
- `GET /models/status` — Model loading state, estimated FPS, and candidate details
- `POST /models/select` — Hot-swap active model (`yolo11n`, `yolo12n`, `yolo26n`, `ensemble`)
- `GET /telemetry` — Live FPS, worker count, violation rate, and compliance percentage
- `POST /api/diagnose` — Single-photo inspection with spatial head-zone bounding boxes & JSON report
- `POST /api/audit/video` — Video clip audit with sampled compliance timeline
- `GET /api/benchmark/baseline` — Empirical comparison metrics against base paper
- `GET /cameras` — Auto-discover connected webcams
- `GET /streams` — Active stream list and state
- `GET /zones` / `PUT /zones` — Configured monitoring zones and PPE compliance rules
- `GET /alerts` — Live compliance violation alerts
- `POST /streams/webcam/start` — Start webcam monitoring
- `POST /streams/video` — Upload and process video file
- `GET /streams/{stream_id}/frame` — Real-time annotated JPEG frame
- `POST /streams/{stream_id}/stop` — Terminate stream

---

## Candidate Models & Hardware Benchmark (VOC2028 Empirical Validation)

| Architecture | Parameters / Weights | Precision | Recall | Macro F1 | mAP50 | mAP50-95 | Hardware Latency (CPU) |
|---|---|---|---|---|---|---|---|
| **Base Paper Benchmark** | Reference | **94.77%** | — | **94.65%** | N/A | N/A | — |
| **YOLO11n** | 5.6 MB | **92.94%** | **86.35%** | **89.52%** | **92.22%** | **59.42%** | 49.1 ms (~35 FPS) |
| **YOLO12n** | 5.6 MB | **84.56%** | **77.55%** | **80.90%** | **83.53%** | **47.02%** | 49.1 ms (~32 FPS) |
| **YOLO26n (Raw)** | 5.5 MB | 2.27% | 9.77% | 3.68% | 0.52% | 0.14% | 36.9 ms (~40 FPS) |
| **Hybrid Ensemble** | Multi-Head Fusion | **94.80%** | **88.07%** | **91.31%** | **94.53%** | **61.08%** | 65.0 ms (~28 FPS) |
