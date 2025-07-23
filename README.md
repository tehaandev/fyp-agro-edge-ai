### 🎓 Final Year Project Summary

**Title**: _Edge-Based Machine Learning Solution for Real-Time Decision Support in Small-Scale Farms_
**Author**: Tehaan Perera
**University**: University of Staffordshire
**Supervisor**: Dr. Tharanga Peiris
**Degree**: BSc (Hons) in Computer Science

---

### 🧩 Problem Statement

Smallholder farms (<2 hectares) produce 30% of global food but lack access to modern tech like ML and IoT. Challenges include:

- Lack of reliable internet (connectivity)
- Cost constraints
- Low technical literacy
  Current DSSs are cloud-based, making them unsuitable for these farms.

---

### 📌 Motivation

- 40% of Sri Lanka's agricultural output is lost due to inefficiencies.
- Even minor improvements in yield drastically improve smallholder livelihoods.

---

### 🔍 Research Gap

- Existing PA systems are too cloud-reliant and resource-heavy.
- Lightweight ML models for edge deployment remain underexplored.

---

### 🎯 Research Aim

To design, implement, and evaluate edge-compatible, offline ML models for smallholder DSS.

---

### 🎯 Research Objectives

1. Design 3 edge-optimized ML pipelines: irrigation, disease detection, crop recommendation.
2. Ensure <1s inference and ≤5W power on Raspberry Pi.

---

### ❓ Research Questions

**Primary**:
How can lightweight ML models be optimized for real-time decision support in resource-constrained smallholder farms?

**Supporting**:

- Which architectures & optimizations best balance size, speed, and accuracy?
- What training strategies work under limited, diverse datasets?
- What are the on-device resource requirements?

---

### 🧪 Methodology

#### Research Onion (Saunders):

- **Philosophy**: Pragmatism
- **Approach**: Deductive
- **Methods**: Mixed (Quant + Qual)
- **Strategies**: Prototyping, Observation, Archival
- **Time Horizon**: Cross-sectional

#### Development Model:

- **RAD (Rapid App Dev)**: Iterative, prototype-driven
- **Languages**: Python (ML), C++ (Microcontroller)
- **Design**: Modular (Sensor → Edge ML → UI)
- **Programming**: Asynchronous event-driven

#### ML Pipeline:

- **Data**: PlantVillage, Kaggle, simulated logs
- **Models**: MobileNetV2 (images), Random Forest/Decision Trees (tabular)
- **Tools**: TensorFlow, TensorFlow Lite, Scikit-learn
- **Optimization**: Quantization, pruning
- **Deployment Target**: Raspberry Pi

---

### 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1
- Inference time
- Memory footprint
- Model size
- Power usage (<5W target)

---

### 🚧 Project Scope

**Included**:

- Dataset collection
- Model design, training, optimization
- TFLite export

**Excluded**:

- Hardware testing (no sensors/RPi integration)
- Usability testing with real farmers
- UI development

---

### 📅 Timeline (10 Weeks)

| Phase                         | Weeks |
| ----------------------------- | ----- |
| Dataset collection & prep     | 1–3   |
| Model training & tuning       | 3–6   |
| Optimization (prune/quantize) | 7–8   |
| Export, document              | 9–10  |

---

### 🔧 Tech Stack

- **Python, C++**
- **TensorFlow, TensorFlow Lite**
- **OpenCV, Scikit-learn**
- **React + Electron (planned UI)**
- **Jupyter/Google Colab**

---

### ⚠️ Risks & Mitigations

| Risk               | Mitigation                       |
| ------------------ | -------------------------------- |
| No proper datasets | Use open datasets + augmentation |
| Model too large    | Quantize & prune early           |
| Low accuracy       | Use transfer learning            |
| Overfitting        | Data augmentation, CV            |

