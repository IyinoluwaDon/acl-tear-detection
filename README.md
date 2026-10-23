# ACL MRI Classification Research Demo

A Streamlit research demo that loads sagittal knee MRI volumes and runs a saved PyTorch classifier for an ACL-tear label. It accepts NumPy volumes and, when optional readers are installed, DICOM and NIfTI files.

## What is included

- MRI volume input validation before inference
- Saved ResNet-based slice-aggregation checkpoint
- Optional DICOM and NIfTI readers

## Run locally

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

## Verify

python -m pip install pytest
python -m pytest -q

## Repository map

`app.py` contains the Streamlit interface, checkpoint loading and scan validation. `best_model.pt` is the local inference checkpoint. `MRNet_ACL_Colab.ipynb` records the experiment; `tests/` covers input validation.

## Evidence and limitations

The repository includes a checkpoint and request/scan-validation tests. The tests verify input contracts; reported historical model metrics have not been independently reproduced in this project.

This is an unvalidated research prototype, not a diagnostic or clinical decision tool. Patient-level and external evaluation, data-use review, and clinical governance are outstanding. Training data is not included; obtain it from its authorized source and follow its terms. Do not upload identifiable patient data to a public deployment.

## License

No repository license file is included. Unless the project owner adds a license, do not assume the code, data or model artifacts may be redistributed or reused. Verify third-party data and model terms separately.
