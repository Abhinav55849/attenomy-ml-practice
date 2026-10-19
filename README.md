# Attenomy : Machine Learning Engineering Internship Starter Repository

Official starter repository, benchmark datasets, and task templates for the **Attenomy ML Engineering Internship Program**.

---

## 🎯 Program Architecture & Structure

This intensive 4-week program is structured into two core phases:

1. **Trial Practice Phase (Weeks 1-2):**
   - **Week 1:** Foundations, Data Handling, Defensive Ingestion, Leak-Free Preprocessing & Statistical EDA.
   - **Week 2:** Classical Machine Learning, Feature Engineering & High-Precision Anti-Fraud Pipeline.
2. **Product Contribution Phase (Weeks 3-4):**
   - **Week 3:** Deep Learning, Vision & NLP Domain Models for **Attenomy Cortex**.
   - **Week 4:** Production Packaging, ONNX Quantization, FastAPI Microservices & Final Capstone Defense.

Official weekly roadmap specifications are available in the [`roadmaps/`](roadmaps/) directory:
- [Week 1 Roadmap PDF](roadmaps/Attenomy_ML_Internship_Week_1_Foundations_and_Data_Engineering.pdf)
- [Week 2 Roadmap PDF](roadmaps/Attenomy_ML_Internship_Week_2_Classical_ML_and_Anti_Fraud_Pipeline.pdf)
- [Week 3 Roadmap PDF](roadmaps/Attenomy_ML_Internship_Week_3_Deep_Learning_and_Domain_Models.pdf)
- [Week 4 Roadmap PDF](roadmaps/Attenomy_ML_Internship_Week_4_Productionization_and_Capstone.pdf)

---

## 📁 Repository Scaffold

```
ml-engineer-internship/
├── data/
│   ├── raw/
│   │   └── attenomy_telemetry_raw.csv    # 25,000-row real-world telemetry dataset
│   ├── clean/                            # Target destination for clean Parquet export
│   └── DATASET_OVERVIEW.md               # 19-column schema data dictionary
├── notebooks/
│   ├── 01_cleaning.ipynb                 # Task 1.2: Guided data cleaning notebook
│   └── 02_eda.ipynb                      # Task 1.3: Guided statistical EDA notebook
├── roadmaps/                             # Complete 4-week official program PDFs
├── reports/
│   ├── figures/                          # Publication-grade chart export directory
│   ├── week1_insights_memo_template.md   # Executive insights memo template
│   └── data_dictionary_template.md       # Data governance schema documentation
├── src/
│   ├── __init__.py
│   ├── utils.py                          # Core data ingestion, health & split utilities
│   └── cleaning.py                       # Standalone CLI cleaning pipeline script
├── tests/
│   ├── __init__.py
│   └── test_utils.py                     # Full pytest test suite
├── LEARNING_LOG.md                       # Daily technical journal template
├── requirements.txt                      # Pinned Python package dependencies
├── pyproject.toml                        # Ruff linter & pytest configurations
└── README.md
```

---

## 🚀 Quick Start & Setup

### 1. Clone the Starter Repository
```bash
git clone https://github.com/attenomy/ml-engineer-internship.git
cd ml-engineer-internship
```

### 2. Configure Python Virtual Environment
**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Test Suite
```bash
pytest -v
```

### 5. Check Code Style & Linting
```bash
ruff check .
```

---

## 📅 Week 1 Milestones & Deliverables

| Task | Component | Key Output | Target Day |
| :--- | :--- | :--- | :--- |
| **Task 1.1** | Repository Scaffold & Utilities | Clean repo, `src/utils.py`, `tests/test_utils.py` | Days 1-2 |
| **Task 1.2** | Data Cleaning & Normalization | `notebooks/01_cleaning.ipynb`, `data/clean/clean.parquet` | Days 2-3 |
| **Task 1.3** | Statistical EDA & Hypothesis Testing | `notebooks/02_eda.ipynb`, `reports/week1_insights_memo.md` | Days 4-5 |

---

## ⏰ Schedule, Deadlines & Mentorship Support

- **Time Commitment:** Part-time flexible (15-20 hours/week).
- **Weekly Submission Deadline:** **Friday by 11:59 PM**. Push all commits, notebooks, and reports to your personal practice repository.
- **Weekend 1-on-1 Feedback Session:** Saturday or Sunday (personalized meeting with an Attenomy ML Lead for code review and feedback).
- **45-Minute Help Escalation Rule:** When encountering a bug or blocker, investigate independently for 45-60 minutes before escalating to your mentor with:
  1. Your objective
  2. The exact error traceback
  3. Steps you already attempted to troubleshoot

---

## 📜 License
Copyright © 2026 Attenomy LLC. Licensed under the Apache License, Version 2.0.
