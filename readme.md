# ⛳ Golf Swing Analysis — VTK Team @ Vietnam Datathon Data Storm 2025

> **AI-powered biomechanics analysis of golf swings** using MediaPipe Pose, Machine Learning, and GPT-4o-mini. Built for the Vietnam Datathon Data Storm 2025 competition.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-green?logo=google)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-black?logo=openai)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This system analyzes golf swing videos from two camera angles (**Back view** and **Side view**), extracts 216 biomechanical features per video using MediaPipe Pose, trains a distance-based scoring model and a PRO vs Amateur classifier, and delivers real-time AI recommendations via a Streamlit web app.

**Competition context:** Vietnam Datathon Data Storm 2025 — Topic: Sports Performance Analysis

---

## 🏗️ Architecture

```
Video Input (MP4/AVI/MOV)
        │
        ▼
MediaPipe Pose Detection
  → 33 body landmarks × (x, y, z, visibility)
        │
        ▼
Feature Engineering (216 features)
  → Joint angles (shoulder, elbow, knee, hip)
  → Normalized distances
  → Joint velocities
  → Statistical summaries (mean, std, skew, kurtosis…)
  → Temporal interpolation → 100 frames
        │
        ├──► Distance-based PRO Scorer  →  Score 0–100 + Level
        │
        └──► PRO vs Amateur Classifier  →  PRO probability
                │
                ▼
         Streamlit Web App
           + GPT-4o-mini AI Recommendations
           + Virtual Coaching Chatbot
```

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📊 **Swing Scoring** | 0–100 score compared against a PRO centroid model |
| 🏆 **Level Classification** | PRO / Advanced / Intermediate / Beginner |
| 📈 **Percentile Ranking** | Where you stand vs the PRO dataset |
| 🔬 **Phase Analysis** | Setup → Backswing → Top → Downswing → Impact |
| 💡 **AI Recommendations** | GPT-4o-mini generates personalized drills with sets/reps/timing |
| 🤖 **Chatbot Coach** | Conversational golf coach with swing context awareness |
| 📐 **Dual View Support** | Back view & Side view models trained separately |

---

## 📁 Project Structure

```
├── scan_data.py                    # Step 0 — Scan raw video dataset, output metadata CSV
├── 01_extract_features_pro.py      # Step 1 — Extract 216 biomechanical features via MediaPipe
├── 02_train_model.py               # Step 2 — Train distance-based PRO scoring model
├── create_amateur_data.py          # Step 3 — Synthesize amateur data (noise + distortion)
├── train_pro_amateur_classifier.py # Step 4 — Train Random Forest PRO vs Amateur classifier
│
├── streamlit_app.py                # Main web application (Streamlit)
├── ai_recommendations.py           # GPT-4o-mini recommendation engine
├── chatbot_assistant.py            # Conversational coaching chatbot
├── chatbot_knowledge.py            # Chatbot knowledge base & FAQ
│
├── models/                         # Trained model files (*.pkl) — see note below
│   ├── scaler_back_v2.pkl
│   ├── model_back_v2.pkl
│   ├── scaler_side_v2.pkl
│   ├── model_side_v2.pkl
│   ├── scaler_classifier_back_v3.pkl
│   ├── classifier_back_v3.pkl
│   ├── scaler_classifier_side_v3.pkl
│   └── classifier_side_v3.pkl
│
├── model_config.json               # Thresholds & config for scoring
├── requirements.txt
├── .env.example                    # Template for API keys
└── README.md
```

> **Note on model files:** `.pkl` files are committed for reproducibility. The PRO training dataset (raw video) is proprietary competition data and is **not included**.

---

## ⚙️ Pipeline — Step by Step

### Step 0 — Scan Dataset
```bash
python scan_data.py
# Output: dataset_metadata_pro.csv
```
Scans all `.mp4` files in the video folder, detects `back`/`side` view from filename prefix, and saves a metadata CSV.

### Step 1 — Extract Features
```bash
python 01_extract_features_pro.py
# Output: features_back_view.csv, features_side_view.csv
```
Runs MediaPipe Pose on each video, skips frames for speed (configurable), resizes to 480p, extracts 216 statistical features per video across 12 biomechanical signals.

**Feature breakdown (216 total):**
- 6 joint angles × 12 stats = 72
- 6 normalized distances × 12 stats = 72
- 6 joint velocities × 12 stats = 72

### Step 2 — Train Scoring Model
```bash
python 02_train_model.py
# Output: scaler_*_v2.pkl, model_*_v2.pkl, model_config.json
```
Builds a **distance-based scoring model** — computes the PRO centroid in standardized feature space. Score = inversely proportional to Mahalanobis-style distance from centroid.

### Step 3 — Synthesize Amateur Data
```bash
python create_amateur_data.py
# Output: features_back_amateur.csv, features_side_amateur.csv
```
Since the competition dataset only contains PRO swings, amateur data is synthesized by adding Gaussian noise, distorting angle features, and randomizing velocity features.

### Step 4 — Train Classifier
```bash
python train_pro_amateur_classifier.py
# Output: scaler_classifier_*_v3.pkl, classifier_*_v3.pkl, classifier_results.json
```
Trains a **Random Forest Classifier** on combined PRO + synthetic amateur data.

### Run the App
```bash
streamlit run streamlit_app.py
```

---

## 🛠️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/golf-swing-analysis.git
cd golf-swing-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run
streamlit run streamlit_app.py
```

---

## 📦 Requirements

```
streamlit
mediapipe
opencv-python
numpy
pandas
scipy
scikit-learn
joblib
openai
python-dotenv
tqdm
```

---

## 🔑 Environment Variables

Create a `.env` file (see `.env.example`):

```env
OPENAI_API_KEY=sk-...
```

The chatbot and AI recommendations require an OpenAI API key. All other analysis features work without it.

---

## 🧠 Key Technical Decisions

**Why distance-based scoring instead of a simple classifier?**
We only had PRO-labeled data. A one-class model (PRO centroid + distance threshold) is more interpretable and generalizes better than a classifier trained only on one class.

**Why 216 features?**
12 statistical descriptors × 18 signals (6 angles + 6 distances + 6 velocities) capture both shape (what the swing looks like) and dynamics (how it moves over time).

**Why synthetic amateur data?**
Competition rules and data availability meant we only had PRO video. Synthetic negative samples (noise + distortion strategies) allowed training a binary classifier as a second scoring signal.

---

## 📊 Model Performance

| View | Classifier Train Acc | Classifier Test Acc |
|------|---------------------|---------------------|
| Back | ~95% | ~88% |
| Side | ~95% | ~87% |

*Results may vary depending on your synthetic data seed.*

---

## 🤝 Team

**VTK Team — Vietnam Datathon Data Storm 2025**

---

## 📄 License

MIT License — free to use for educational and portfolio purposes.
