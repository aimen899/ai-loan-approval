# 🏦 LoanIQ — AI Loan Approval Predictor

A production-grade ML web app that predicts loan approval status
using a Random Forest classifier trained on 4,269 records with 98.1% accuracy.

---

## 📁 Folder Structure

```
loan_app/
│
├── app.py                   ← Main Streamlit application
├── requirements.txt         ← Python dependencies
├── save_model_colab.py      ← Run this in Colab to export models
├── README.md                ← This file
│
├── .streamlit/
│   └── config.toml          ← App theme configuration
│
├── loan_model.pkl           ← (You generate this from Colab)
├── preprocessor.pkl         ← (You generate this from Colab)
└── label_encoder.pkl        ← (You generate this from Colab)
```

---

## 🚀 Step-by-Step Setup Guide

### STEP 1 — Export Model from Google Colab

1. Open your existing Colab notebook
2. Scroll to the very end (after all Phase 4 code)
3. Add a new cell and paste the contents of `save_model_colab.py`
4. Run that cell
5. Three files will automatically download to your computer:
   - `loan_model.pkl`
   - `preprocessor.pkl`
   - `label_encoder.pkl`
6. Move all three `.pkl` files into the `loan_app/` folder

---

### STEP 2 — Install Python (if not already installed)

Download from: https://www.python.org/downloads/
✅ Check "Add Python to PATH" during installation

---

### STEP 3 — Open Terminal / Command Prompt

**Windows:** Press `Win + R`, type `cmd`, press Enter

**Mac:** Press `Cmd + Space`, type `terminal`, press Enter

Navigate to your loan_app folder:
```bash
cd path/to/loan_app
# Example Windows: cd C:\Users\YourName\Desktop\loan_app
# Example Mac:     cd ~/Desktop/loan_app
```

---

### STEP 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

Wait for all packages to install (takes 2-3 minutes).

---

### STEP 5 — Run the App

```bash
streamlit run app.py
```

Your browser will automatically open at:
```
http://localhost:8501
```

The app is now running! 🎉

---

### STEP 6 — Deploy Online (Free) via Streamlit Cloud

To get a public link anyone can access:

1. Create a free account at **https://github.com** (if you don't have one)

2. Create a new repository called `loan-approval-app`

3. Upload all files from `loan_app/` to the repository
   (including the `.streamlit/` folder and all `.pkl` files)

4. Go to **https://share.streamlit.io**

5. Sign in with your GitHub account

6. Click **"New app"**

7. Select your repository → Branch: `main` → Main file: `app.py`

8. Click **"Deploy"**

9. Wait 2-3 minutes — you get a public link like:
   ```
   https://your-name-loan-approval-app.streamlit.app
   ```

Share this link in your project report for full extra credit! ✅

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: streamlit` | Run `pip install streamlit` again |
| `FileNotFoundError: loan_model.pkl` | Make sure .pkl files are in same folder as app.py |
| `Port 8501 already in use` | Run `streamlit run app.py --server.port 8502` |
| Browser doesn't open | Manually go to `http://localhost:8501` |
| sklearn version mismatch | Re-export .pkl files using same sklearn version |

---

## 📊 Model Details

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training Samples | 3,415 |
| Test Samples | 854 |
| Accuracy | 98.1% |
| ROC-AUC | 0.99 |
| Top Feature | CIBIL Score (79.5% importance) |
| Classes | Approved / Rejected |

---

## 👥 Team

Loan Approval Classification Project
Machine Learning Course — Academic Submission
