# 🌸 Flowers Classification

5-class flower image classification using **MobileNetV2** transfer learning.

---

## Model
- **Base:** MobileNetV2 (frozen, ImageNet weights)
- **Head:** GlobalAvgPool → Dense(128, relu, L2) → BatchNorm → Dropout(0.5) → Dense(5, softmax)
- **Input:** 128 × 128 × 3
- **Loss:** sparse_categorical_crossentropy

---

## Dataset
- **Classes:** Daisy · Dandelion · Rose · Sunflower · Tulip
- **Total Images:** 4,317
- **Split:** 3,454 train / 863 validation
- **Batch size:** 32

---

## Results
| Epoch | Train Acc | Val Acc |
|:---:|:---:|:---:|
| 1 | 73.62% | 84.59% |
| 9 | 91.23% | 87.02% |
| 15 | 92.47% | 85.17% |

---

## Tech Stack
```
TensorFlow 2.21.0 | Streamlit >=1.35.0 | NumPy | Pillow
```

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
