# 🎯 NEST Prediction System

> An AI-powered prediction engine for the **National Entrance Screening Test (NEST)** that forecasts high-probability topics and question patterns using historical exam data and an ensemble of machine learning models.

🌐 **Live Demo:** [nest-prediction.vercel.app](https://nest-prediction.vercel.app)

---

## 📌 Overview

The NEST Prediction System analyzes years of NEST exam data to identify recurring topic patterns, subject-wise distributions, and question trends. It leverages a multi-model ML ensemble (XGBoost, LightGBM, CatBoost, Random Forest, and more) to generate confident predictions for upcoming NEST exams — helping aspirants focus their preparation strategically.

---

## ✨ Features

- **Multi-subject coverage** — Physics, Chemistry, Biology, Mathematics, and General Aptitude
- **Elite ML ensemble** — Stacks XGBoost, LightGBM, CatBoost, ExtraTrees, and Bayesian Ridge for robust predictions
- **Era-weighted analysis** — Assigns higher importance to recent papers (2023+) while preserving historical signal
- **Topic normalization** — Automatically merges duplicate/fragmented topic variants into canonical names
- **Syllabus-aware filtering** — Validates topics against the official NEST syllabus
- **Multi-shift support** — Handles years with multiple exam shifts (2020, 2022, 2023, 2024)
- **Confidence scoring** — Each prediction comes with a calibrated confidence score
- **Interactive frontend** — Browse predictions by subject and year through a clean web UI
- **REST API** — Programmatic access to predictions via a JSON API

---

## 🗂️ Project Structure

```
NEST-prediction/
│
├── nest_prediction_system.py      # Core ML prediction engine (v3.2)
├── nest_prediction_new.py         # Updated prediction pipeline
├── gate_prediction_system.py      # Original GATE system (adapted for NEST)
│
├── api/                           # Backend API (served to frontend)
│
├── index.html                     # Frontend web interface
├── data.js                        # Frontend data layer
│
├── all_merged_cleaned.json        # Cleaned historical NEST question bank
├── nest2026_predictions.json      # Generated predictions for NEST 2026
├── nest2026_predictions_v6.json   # v6 refined predictions
├── nest2026_evaluation.json       # Evaluation metrics for predictions
├── nest_elite_predictions_2027.json  # Forward-looking 2027 predictions
│
├── catboost_info/                 # CatBoost training logs
├── __pycache__/                   # Python bytecode cache
└── .env                           # Environment variables
```

---

## 🧠 How It Works

### 1. Data Loading & Processing
Historical NEST questions are loaded from `all_merged_cleaned.json`. Each question is validated, normalized, and tagged with:
- Subject (Physics / Chemistry / Biology / Mathematics / General)
- Chapter (topic) and micro-topic
- Year and shift
- Era weight (recent years get higher weight)
- Syllabus penalty (off-syllabus topics are down-weighted)

### 2. Feature Engineering
For each topic, 55 features are extracted across 6 categories:

| Feature Group | Examples |
|---|---|
| **Temporal** | Weighted mean, EMA (5 alphas), trend slope, R², momentum |
| **Statistical** | Min, max, IQR, CV, skewness, kurtosis |
| **Era-based** | Per-era averages (Early / Stable / Recent / Current) |
| **Consistency** | Appearance rate, max consecutive years, stability score |
| **Syllabus-aware** | Core topic flag, subject quota, percentile in subject |
| **Recency** | Last 5-year counts, years since last seen |

### 3. Ensemble Prediction
Available models are stacked:
- XGBoost
- LightGBM
- CatBoost
- Random Forest / Extra Trees
- Histogram Gradient Boosting
- Bayesian Ridge / Huber Regression (for robustness)

Predictions are calibrated, confidence-scored, and filtered by a minimum threshold (`0.15`).

### 4. Output
Predictions are written to JSON files and served via the API and frontend.

---

## 📊 Subject Distribution (per paper)

| Subject | Questions | Share |
|---|---|---|
| Mathematics | 20 | 25% |
| Physics | 20 | 25% |
| Biology | 19 | 24% |
| Chemistry | 18 | 23% |
| General | 3 | ~4% |
| **Total** | **80** | **100%** |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js (for the frontend/API)

### Installation

```bash
# Clone the repository
git clone https://github.com/RudraPratapSingh2008/NEST-prediction.git
cd NEST-prediction

# Install Python dependencies
pip install numpy pandas scikit-learn xgboost lightgbm catboost scipy
```

### Run the Prediction Engine

```bash
python nest_prediction_system.py
```

This will:
1. Load and validate `all_merged_cleaned.json`
2. Engineer features for all topics
3. Train the ML ensemble
4. Output predictions to `nest2026_predictions.json`

### View the Frontend

Open `index.html` in your browser, or deploy to Vercel (as configured).

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `numpy` / `pandas` | Data manipulation |
| `scikit-learn` | ML models, preprocessing, stacking |
| `xgboost` | Gradient boosting |
| `lightgbm` | Fast gradient boosting |
| `catboost` | Categorical boosting |
| `scipy` | Statistical functions |

All ML libraries degrade gracefully — the system runs with whatever subset is installed.

---

## 📁 Data Format

Each question in `all_merged_cleaned.json` follows this schema:

```json
{
  "year": 2023,
  "shift": "Shift 1",
  "subject": "Physics",
  "chapter": "Optics",
  "topic": "Interference",
  "question_text": "...",
  "choices": { "A": "...", "B": "...", "C": "...", "D": "..." },
  "answer": "B",
  "marks": 3
}
```

---

## 🔮 Prediction Output Format

```json
{
  "subject": "Physics",
  "topic": "Optics",
  "predicted_questions": 3,
  "confidence": 0.87,
  "trend": "increasing",
  "last_seen": 2024,
  "years_active": 8
}
```

---

## ⚙️ Configuration

Key constants in `nest_prediction_system.py`:

```python
TOTAL_NEST_QUESTIONS = 80          # Questions per paper
MIN_APPEARANCES = 2                # Minimum topic appearances to include
MIN_CONFIDENCE_THRESHOLD = 0.15    # Confidence cutoff for predictions
DIAGNOSTIC_MODE = True             # Verbose logging
```

Era weights (how much each period contributes):

```python
ERA_WEIGHTS = {
    "ancient": 0.05,   # Pre-2011
    "early":   0.30,   # 2011–2014
    "stable":  0.65,   # 2015–2019
    "recent":  0.85,   # 2020–2022
    "current": 1.00    # 2023+
}
```

---

## 🤝 Contributing

Contributions are welcome! If you have updated question data, improved models, or UI enhancements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source. See the repository for details.

---

## 👤 Author

**Rudra Pratap Singh**
GitHub: [@RudraPratapSingh2008](https://github.com/RudraPratapSingh2008)

---

> ⚠️ **Disclaimer:** Predictions are based on historical patterns and ML models. They are intended to guide preparation, not guarantee exam outcomes. Always cover the full syllabus.
