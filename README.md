# ACL Tear Detection — MRI Classification System

An AI-powered web application that analyses knee MRI scans and detects ACL (Anterior Cruciate Ligament) tears using deep learning.

---

## Model Performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.9719 |
| Sensitivity | 0.7778 |
| Specificity | 0.9394 |
| Accuracy | 0.87 |

---

## Supported File Formats

| Format | Description |
|---|---|
| `.npy` | MRNet research format (NumPy array) |
| `.dcm` | DICOM — hospital standard |
| `.nii` / `.nii.gz` | NIfTI — clinical/research format |

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/your-username/acl-tear-detection
cd acl-tear-detection

# Install dependencies
pip install -r requirements.txt

# Add your trained model weights
# Place best_model.pt in the root directory

# Run the app
streamlit run app.py
```

---

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (include `best_model.pt`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py`
5. Deploy — you get a public link instantly

---

## Architecture

- **Backbone:** ResNet18 pretrained on ImageNet (transfer learning)
- **Aggregation:** Max pooling across MRI slices
- **Dataset:** Stanford MRNet — 1,130 training exams, sagittal plane
- **Framework:** PyTorch

---

## Clinical Disclaimer

This tool is an AI-assisted decision support system. It is not a substitute for professional medical diagnosis. All results must be reviewed and confirmed by a qualified radiologist or orthopaedic clinician before any clinical decision is made.
