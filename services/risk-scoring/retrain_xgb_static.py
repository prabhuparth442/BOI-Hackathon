"""
BOI Sentinel AI — XGBoost Retrainer (Static-Feature Aligned)
=============================================================
PROBLEM: The current model was trained on CICMalDroid's 470-column
syscall/binder frequency CSV.  At inference the pipeline feeds it
12 security-indicator features from Androguard + Frida (permission
counts, YARA hits, obfuscation flags …).  The distributions don't
match, so the model hallucinates scores and needs a hardcoded cap.

FIX:  Retrain on CICMalDroid's **50,621-column static CSV** which
contains permissions, intents, sensitive APIs, services, receivers —
the same things Androguard extracts at runtime.

USAGE (Google Colab):
    1. Upload the 50,621-feature static CSV from CICMalDroid 2020
       (download from https://www.unb.ca/cic/datasets/maldroid-2020.html)
    2. Run this script cell-by-cell.
    3. Download the output model and drop it into services/risk-scoring/models/

Author: retrain helper for Samarth / BOI Hackathon
"""

# =====================================================================
# CELL 0 — Install deps (run once)
# =====================================================================
# !pip install xgboost scikit-learn shap pandas numpy joblib

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os, warnings
warnings.filterwarnings("ignore")


# =====================================================================
# CELL 1 — Load the static CSV
# =====================================================================
# UPDATE THIS PATH to wherever you placed the 50,621-feature static CSV.
# In Colab: upload via files.upload() or mount Google Drive.
STATIC_CSV = "/content/total_features.csv"          # <-- ADJUST

# If the file is inside a zip, unzip first:
# !unzip -o "/content/CSV.zip" -d /content/csv_extracted/
# Then point STATIC_CSV at the right file inside csv_extracted/.

df = pd.read_csv(STATIC_CSV, low_memory=False, encoding="latin-1")
print(f"Loaded: {df.shape[0]} samples × {df.shape[1]} columns")
print(f"Columns (first 30): {list(df.columns[:30])}")
print(f"Columns (last 30):  {list(df.columns[-30:])}")

# Identify the class/label column
POSSIBLE_LABEL_COLS = ["Class", "class", "Label", "label", "Category", "category"]
label_col = None
for c in POSSIBLE_LABEL_COLS:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    # Sometimes the label is the last column
    label_col = df.columns[-1]
    print(f"WARNING: Guessing label column is '{label_col}'")
else:
    print(f"Label column: '{label_col}'")

print(f"\nLabel distribution:\n{df[label_col].value_counts()}")


# =====================================================================
# CELL 2 — Map class labels to the 5-class scheme
# =====================================================================
# CICMalDroid uses integer labels 1-5 OR string names.
# Adapt this mapping based on what you see in the distribution above.

CLASS_MAP_INT = {
    1: "Adware",
    2: "Banking",
    3: "SMS",
    4: "Riskware",
    5: "Benign",
}
CLASS_MAP_STR = {
    "adware": "Adware",
    "banking": "Banking",
    "sms": "SMS",
    "riskware": "Riskware",
    "benign": "Benign",
}

CLASSES = ["Benign", "Riskware", "Adware", "SMS", "Banking"]  # keep this order — matches inference
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
SEVERITY_VEC = np.array([5, 35, 55, 75, 95], dtype=np.float32)

# Try integer mapping first, then string
labels_raw = df[label_col]
if labels_raw.dtype in [np.int64, np.float64, int, float]:
    df["CategoryName"] = labels_raw.astype(int).map(CLASS_MAP_INT)
else:
    df["CategoryName"] = labels_raw.str.strip().str.lower().map(CLASS_MAP_STR)

# Drop any rows that didn't map (edge-case bad rows)
unmapped = df["CategoryName"].isna().sum()
if unmapped > 0:
    print(f"WARNING: {unmapped} rows had unmapped labels — dropping them.")
    df = df.dropna(subset=["CategoryName"])

print(f"\nFinal label distribution:\n{df['CategoryName'].value_counts()}")


# =====================================================================
# CELL 3 — Extract the 12 features that MATCH the inference pipeline
# =====================================================================
# The inference pipeline (feature_extractor.py) produces these 12:
#   dangerous_perm_count, suspicious_api_count, yara_match_count,
#   obfuscation_detected, dynamic_code_loading, hardcoded_url_count,
#   malicious_ioc_count, sms_intercepted, accessibility_abuse,
#   c2_connection_count, runtime_downloads, ai_confidence
#
# The 50,621-column static CSV has columns for individual permissions,
# intents, APIs, services, etc.  We sum/flag them to match.

FEATURE_NAMES = [
    "dangerous_perm_count",
    "suspicious_api_count",
    "yara_match_count",
    "obfuscation_detected",
    "dynamic_code_loading",
    "hardcoded_url_count",
    "malicious_ioc_count",
    "sms_intercepted",
    "accessibility_abuse",
    "c2_connection_count",
    "runtime_downloads",
    "ai_confidence",
]

# --- Helper: find columns matching ANY of the keywords (case-insensitive) ---
def find_cols(df, keywords):
    """Return list of column names that contain any keyword (case-insensitive)."""
    cols_lower = {c: c.lower() for c in df.columns}
    matched = []
    for col, col_low in cols_lower.items():
        if any(kw.lower() in col_low for kw in keywords):
            matched.append(col)
    return matched

def sum_binary(df, cols):
    """Sum columns (treating non-numeric as 0) across matched columns."""
    if not cols:
        return pd.Series(0, index=df.index)
    subset = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return subset.sum(axis=1)

def any_present(df, cols):
    """1.0 if any matched column has a non-zero value, else 0.0."""
    if not cols:
        return pd.Series(0.0, index=df.index)
    subset = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return (subset.sum(axis=1) > 0).astype(float)


print("Mapping 50k static columns → 12 inference features...\n")

features = pd.DataFrame(index=df.index)

# 1. dangerous_perm_count — sum of dangerous permission columns
DANGEROUS_PERM_KEYWORDS = [
    "READ_SMS", "RECEIVE_SMS", "SEND_SMS",
    "READ_CONTACTS", "WRITE_CONTACTS",
    "RECORD_AUDIO", "CAMERA",
    "READ_CALL_LOG", "WRITE_CALL_LOG",
    "BIND_ACCESSIBILITY_SERVICE", "BIND_DEVICE_ADMIN",
    "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE",
    "GET_ACCOUNTS", "USE_CREDENTIALS",
    "SYSTEM_ALERT_WINDOW",
    "RECEIVE_BOOT_COMPLETED",
    "REQUEST_INSTALL_PACKAGES",
]
perm_cols = find_cols(df, DANGEROUS_PERM_KEYWORDS)
features["dangerous_perm_count"] = sum_binary(df, perm_cols)
print(f"  dangerous_perm_count: matched {len(perm_cols)} columns")

# 2. suspicious_api_count — sensitive/suspicious API invocations
SUSPICIOUS_API_KEYWORDS = [
    "getDeviceId", "getSubscriberId", "getImei", "getSimSerialNumber",
    "sendTextMessage", "sendMultipartTextMessage",
    "execCommand", "Runtime", "getRuntime",
    "DexClassLoader", "PathClassLoader", "InMemoryDexClassLoader",
    "AccessibilityService", "performGlobalAction",
    "getInstalledPackages", "getInstalledApplications",
    "Cipher", "SecretKeySpec",
    "Base64", "getDeclaredMethod",
    "sensitive_api",  # CICMalDroid often has aggregated sensitive_api columns
]
api_cols = find_cols(df, SUSPICIOUS_API_KEYWORDS)
features["suspicious_api_count"] = sum_binary(df, api_cols)
print(f"  suspicious_api_count: matched {len(api_cols)} columns")

# 3. yara_match_count — not in static CSV; set to 0 (same as before, but now
#    the model won't rely on it since other features are properly mapped)
features["yara_match_count"] = 0
print(f"  yara_match_count: set to 0 (not in dataset, comes from YARA at runtime)")

# 4. obfuscation_detected — look for obfuscation-related columns
obf_cols = find_cols(df, ["obfuscat", "proguard", "encrypt", "pack"])
features["obfuscation_detected"] = any_present(df, obf_cols)
print(f"  obfuscation_detected: matched {len(obf_cols)} columns")

# 5. dynamic_code_loading — DexClassLoader, reflection, etc.
dcl_cols = find_cols(df, [
    "DexClassLoader", "PathClassLoader", "ClassLoader",
    "loadClass", "defineClass", "loadDex",
    "dalvik.system",
])
features["dynamic_code_loading"] = any_present(df, dcl_cols)
print(f"  dynamic_code_loading: matched {len(dcl_cols)} columns")

# 6. hardcoded_url_count — URL/network-related columns
url_cols = find_cols(df, ["url", "http", "https", "ftp", "URI"])
features["hardcoded_url_count"] = sum_binary(df, url_cols)
print(f"  hardcoded_url_count: matched {len(url_cols)} columns")

# 7. malicious_ioc_count — no ground truth IOC data in static CSV
features["malicious_ioc_count"] = 0
print(f"  malicious_ioc_count: set to 0 (comes from threat-intel service at runtime)")

# 8. sms_intercepted — SMS-related permissions/APIs present
sms_cols = find_cols(df, [
    "READ_SMS", "RECEIVE_SMS", "SEND_SMS",
    "sendTextMessage", "SmsManager",
    "android.provider.Telephony.SMS_RECEIVED",
])
features["sms_intercepted"] = any_present(df, sms_cols)
print(f"  sms_intercepted: matched {len(sms_cols)} columns")

# 9. accessibility_abuse — accessibility service abuse
acc_cols = find_cols(df, [
    "BIND_ACCESSIBILITY_SERVICE",
    "AccessibilityService", "AccessibilityEvent",
    "performGlobalAction",
])
features["accessibility_abuse"] = any_present(df, acc_cols)
print(f"  accessibility_abuse: matched {len(acc_cols)} columns")

# 10. c2_connection_count — network/socket/connection columns
c2_cols = find_cols(df, [
    "INTERNET", "ACCESS_NETWORK_STATE",
    "socket", "connect", "HttpURLConnection",
    "network", "send", "recv",
])
# Use a softer count here (not just binary)
features["c2_connection_count"] = sum_binary(df, c2_cols).clip(upper=20)
print(f"  c2_connection_count: matched {len(c2_cols)} columns")

# 11. runtime_downloads — file/download related
dl_cols = find_cols(df, [
    "WRITE_EXTERNAL", "download", "DownloadManager",
    "openConnection", "InputStream",
])
features["runtime_downloads"] = sum_binary(df, dl_cols).clip(upper=10)
print(f"  runtime_downloads: matched {len(dl_cols)} columns")

# 12. ai_confidence — placeholder (comes from LLM agent at inference)
#     Use 0.5 as neutral; the model learns it's uninformative
features["ai_confidence"] = 0.5
print(f"  ai_confidence: set to 0.5 (comes from AI agent at runtime)")

features = features[FEATURE_NAMES]  # enforce column order
print(f"\nFinal feature matrix: {features.shape}")
print(features.describe().round(2))


# =====================================================================
# CELL 4 — Sanity checks before training
# =====================================================================
# Verify feature distributions look reasonable
print("\n=== Feature distribution per class ===")
features["_label"] = df["CategoryName"].values
for cls in CLASSES:
    subset = features[features["_label"] == cls]
    print(f"\n{cls} ({len(subset)} samples):")
    print(subset[FEATURE_NAMES].mean().round(2).to_string())
features.drop("_label", axis=1, inplace=True)

# Check for features with zero variance (useless for the model)
zero_var = [f for f in FEATURE_NAMES if features[f].std() == 0]
if zero_var:
    print(f"\n⚠️  Zero-variance features (model can't learn from these): {zero_var}")
    print("   These features will still be in the model for compatibility,")
    print("   but won't contribute to predictions. They'll get real values at inference.")
else:
    print("\n✅ All features have non-zero variance.")


# =====================================================================
# CELL 5 — Train XGBoost
# =====================================================================
y = df["CategoryName"].map(class_to_idx).astype(int).values
X = features.astype(np.float32).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

w_train = compute_sample_weight("balanced", y_train)

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, feature_names=FEATURE_NAMES)
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=FEATURE_NAMES)

params = {
    "objective":       "multi:softprob",
    "num_class":       len(CLASSES),
    "max_depth":       6,
    "eta":             0.1,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "eval_metric":     "mlogloss",
    "tree_method":     "hist",       # fast on CPU
    "seed":            42,
}

print("\nTraining XGBoost...")
model = xgb.train(
    params, dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=30,
    verbose_eval=25,
)

# Evaluate
proba    = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
pred_idx = proba.argmax(axis=1)
print(f"\nCategory accuracy: {accuracy_score(y_test, pred_idx):.2%}")
print(classification_report(y_test, pred_idx, target_names=CLASSES))

# Risk score distribution
risk_scores = proba @ SEVERITY_VEC
print(f"\nRisk score stats: mean={risk_scores.mean():.1f}, "
      f"median={np.median(risk_scores):.1f}, "
      f"min={risk_scores.min():.1f}, max={risk_scores.max():.1f}")

# Verify benign apps get low scores, banking malware gets high scores
for cls_name in CLASSES:
    mask = y_test == class_to_idx[cls_name]
    if mask.sum() > 0:
        cls_scores = risk_scores[mask]
        print(f"  {cls_name:12s}: mean={cls_scores.mean():.1f}, "
              f"median={np.median(cls_scores):.1f}, "
              f"range=[{cls_scores.min():.1f}, {cls_scores.max():.1f}]")


# =====================================================================
# CELL 6 — Confusion Matrix (visual sanity check)
# =====================================================================
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, pred_idx)
# Pretty print
header = "  " + "  ".join(f"{c[:4]:>6}" for c in CLASSES)
print(header)
for i, row in enumerate(cm):
    vals = "  ".join(f"{v:6d}" for v in row)
    print(f"{CLASSES[i][:4]:>6}: {vals}")


# =====================================================================
# CELL 7 — Save the model
# =====================================================================
os.makedirs("/content/output", exist_ok=True)

# Save as JSON (for xgb.Booster.load_model — what main.py uses)
MODEL_JSON = "/content/output/xgb_risk_model.json"
model.save_model(MODEL_JSON)
print(f"\nSaved JSON model: {MODEL_JSON}")

# Also save as pkl (backup)
MODEL_PKL = "/content/output/xgb_risk_model.pkl"
joblib.dump(model, MODEL_PKL)
print(f"Saved PKL model:  {MODEL_PKL}")

print(f"\n{'='*60}")
print("DONE!  Next steps:")
print("  1. Download xgb_risk_model.json")
print("  2. Replace services/risk-scoring/models/xgb_risk_model.json")
print("  3. (Optional) Also replace xgb_risk_model.pkl")
print("  4. Remove the safety cap in model.py (lines 70-88)")
print("     — the retrained model shouldn't need it")
print(f"{'='*60}")

# Auto-download in Colab
try:
    from google.colab import files
    files.download(MODEL_JSON)
    files.download(MODEL_PKL)
except ImportError:
    pass
