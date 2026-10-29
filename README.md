# ACL Tear Classification from Knee MRI Volumes

This project trains and serves a binary ACL-tear classifier on sagittal knee MRI volumes from the Stanford MRNet dataset. The Streamlit interface is the inference front end to a PyTorch model whose training workflow and evaluation are recorded in `MRNet_ACL_Colab.ipynb`.

## Model and training

The network uses an ImageNet-pretrained ResNet18 as a per-slice feature extractor. Each sagittal slice is min-max normalized, resized to 224 x 224, and replicated across three channels. The 512-dimensional ResNet feature vectors are max-pooled across the scan's variable number of slices, then passed through a 512->256->64->1 fully connected classifier with ReLU activations and 0.3 dropout.

Training uses binary cross-entropy with logits and a positive-class weight, Adam (learning rate 1e-4, weight decay 1e-5), a validation-AUC learning-rate scheduler, and early stopping (up to 30 epochs, patience 7). Training augmentation applies horizontal flips and rotations. The notebook reports a validation ROC AUC of **0.9719** for its selected checkpoint.

## Data and files

The training source is Stanford's MRNet knee MRI dataset. The dataset is not
stored in this repository; place the `MRNet-v1.0` sagittal `.npy` volumes and
ACL label CSVs in the paths documented by the notebook before retraining.
`best_model.pt` is the checkpoint loaded by the app. `app.py` handles
inference, scan preprocessing, and optional DICOM/NIfTI/NumPy uploads;
`scan_validation.py` checks volume structure; `tests/` covers the input
contract.

## Run

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

The local app runs inference from `best_model.pt`; rerun the notebook to reproduce training and its reported evaluation.
