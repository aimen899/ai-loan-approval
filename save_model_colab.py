# ============================================================
# STEP 1 — Run this code at the END of your Colab notebook
# It saves your trained model and preprocessor to disk
# then downloads them to your computer
# ============================================================

import joblib
from google.colab import files

# Save the best performing model (Random Forest)
joblib.dump(rf_model, 'loan_model.pkl')

# Save the preprocessor pipeline (scaler + encoder)
joblib.dump(preprocessor, 'preprocessor.pkl')

# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')

print("✅ Models saved successfully!")
print("📥 Downloading files to your computer...")

# Download all 3 files
files.download('loan_model.pkl')
files.download('preprocessor.pkl')
files.download('label_encoder.pkl')

print("✅ Done! Place all 3 .pkl files in your loan_app folder.")
