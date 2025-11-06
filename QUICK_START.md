# Quick Start Guide

## ⚡ 30-Second Status

**What's Done**: 2 ML models trained and ready
**What's Next**: Integrate irrigation model into POC notebook
**Time Needed**: ~1 hour
**Files to Edit**: `notebooks/0_poc.ipynb`

---

## 🎯 Your Next Task

### Open this file:
```bash
notebooks/0_poc.ipynb
```

### Add a new section called:
```markdown
## Irrigation Control Demo (RandomForest)
```

### Copy-paste this code:
```python
import pickle
import numpy as np

# Load model and artifacts
with open('../models/irrigation_rf_final.pkl', 'rb') as f:
    irrigation_model = pickle.load(f)

with open('../data/processed/irrigation/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load test data
X_test = np.load('../data/processed/irrigation/X_test_flat.npy')
y_test = np.load('../data/processed/irrigation/y_test.npy')

# Test prediction
sample_idx = 100
prediction = irrigation_model.predict(X_test[sample_idx:sample_idx+1])[0]
probabilities = irrigation_model.predict_proba(X_test[sample_idx:sample_idx+1])[0]

print(f"Predicted: {prediction}")
print(f"Actual: {y_test[sample_idx]}")
print(f"Probabilities: {dict(zip(['Irrigate_High', 'Irrigate_Low', 'No_Irrigation'], probabilities))}")
```

### Run it and verify:
- ✅ Model loads successfully
- ✅ Prediction matches actual label (or close)
- ✅ Probabilities sum to 1.0

---

## 📚 Context Documents (Read if Needed)

1. **`CLAUDE_CODE_CONTEXT.md`** - Quick context (5 min read)
2. **`DEVELOPMENT_HANDOFF.md`** - Full context (15 min read)
3. **`docs/IRRIGATION_MODEL_SUMMARY.md`** - Model reference

---

## 🎓 Why RandomForest? (For Defense)

**Question**: "Why 97.89% instead of 100%?"
**Answer**: "DecisionTree's 100% indicates overfitting. RandomForest's 97.89% is more realistic for production sensor data."

**Evidence**: `docs/MODEL_SELECTION_RATIONALE.md`

---

## ✅ Current Status

- ✅ Disease detection model: 98% accuracy, 9.1 MB
- ✅ Irrigation control model: 97.89% accuracy, 50 KB
- ✅ Both tested and validated
- 🎯 **Next**: Combine in POC notebook

---

**Start Here**: Open `notebooks/0_poc.ipynb` and add irrigation section
**Questions?**: Read `CLAUDE_CODE_CONTEXT.md` for details
