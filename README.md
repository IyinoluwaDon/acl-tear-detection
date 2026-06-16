# ACL Tear Detection — MRI Classification System

An AI-powered web application that analyses knee MRI scans and detects ACL (Anterior Cruciate Ligament) tears using deep learning trained on the Stanford MRNet dataset.

---

## Live Demo

> https://acl-tear-detection.streamlit.app/

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

## Dataset

This model was trained on the **Stanford MRNet Dataset** — a dataset of 1,370 knee MRI exams collected at Stanford University Medical Center.

| Detail | Info |
|---|---|
| Total exams | 1,370 |
| Training set | 1,130 exams |
| Validation set | 120 exams |
| Plane used | Sagittal only |
| Labels used | ACL tear (binary) |

### How to Access the Data

The dataset is not included in this repository. To obtain it:

1. **Official source — Stanford AIMI:**
   ```
   https://aimi.stanford.edu/datasets/mrnet-knee-mris
   ```
   Register with an institutional or personal email. Access is granted after agreeing to a research use agreement.

2. **Original Stanford ML Group page:**
   ```
   https://stanfordmlgroup.github.io/competitions/mrnet/
   ```
   Fill the registration form at the bottom of the page. A download link is emailed to you immediately.

3. **Kaggle mirror** (fastest, no wait):
   ```
   https://www.kaggle.com/datasets?search=MRNet+Stanford+knee
   ```
   Search for MRNet on Kaggle. Several community uploads exist with identical data.

> **Note:** This project uses only the sagittal plane and ACL tear labels. You do not need to download the full 7.7GB dataset — the sagittal folder (~500MB) and `train-acl.csv` / `valid-acl.csv` are sufficient.

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/IyinoluwaDon/acl-tear-detection
cd acl-tear-detection

# Install dependencies
pip install -r requirements.txt

# Add your trained model weights
# Place best_model.pt in the root directory
# Train your own using the notebook, or request access from the repo owner

# Run the app
streamlit run app.py
```

---

## Training Your Own Model

The training notebook is included in this repository. To retrain:

1. Download the MRNet dataset from one of the sources above
2. Extract only the sagittal plane folders and ACL CSV labels
3. Structure your data as:
   ```
   MRNet_ACL/
   ├── train-acl.csv
   ├── valid-acl.csv
   └── MRNet-v1.0/
       ├── train/sagittal/
       └── valid/sagittal/
   ```
4. Run the MRNet_ACL_Colab.ipynb file and edit file path consistency
5. Measure metrics 

---

## Deployment (Streamlit Community Cloud)

1. Fork or clone this repo to your GitHub account
2. Add your `best_model.pt` to the root directory (use Git LFS for files over 100MB)
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repo
5. Set main file to `app.py`
6. Deploy — you get a public link instantly

---

## Architecture

- **Backbone:** ResNet18 pretrained on ImageNet (transfer learning)
- **Input:** Sagittal knee MRI — variable number of slices per scan
- **Slice processing:** Each slice processed independently through ResNet18
- **Aggregation:** Max pooling across all slices → single 512-dim feature vector
- **Head:** Fully connected layers → single logit → sigmoid → ACL tear probability
- **Loss:** BCEWithLogitsLoss with pos_weight for class imbalance correction
- **Framework:** PyTorch

---

## References

- Bien, N., Rajpurkar, P., Ball, R. L., et al. (2018). *Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet.* PLOS Medicine. https://doi.org/10.1371/journal.pmed.1002686
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR. https://arxiv.org/abs/1512.03385

---

## Clinical Disclaimer

This tool is an AI-assisted decision support system. It is **not** a substitute for professional medical diagnosis. All results must be reviewed and confirmed by a qualified radiologist or orthopaedic clinician before any clinical decision is made.
