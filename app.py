import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import tempfile

# ── Optional imports for DICOM and NIfTI ─────────────────────
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

try:
    import nibabel as nib
    NIFTI_AVAILABLE = True
except ImportError:
    NIFTI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
# Must match exactly what was used during training
# ─────────────────────────────────────────────────────────────

class MRNetModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(MRNetModel, self).__init__()
        resnet = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, num_slices, C, H, W = x.shape
        x        = x.view(batch_size * num_slices, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_slices, 512)
        features, _ = torch.max(features, dim=1)
        return self.classifier(features)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load trained model once and cache it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MRNetModel(dropout_rate=0.3).to(device)

    checkpoint = torch.load(
        "best_model.pt",
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


# ─────────────────────────────────────────────────────────────
# FILE LOADING — supports .npy, .dcm, .nii / .nii.gz
# ─────────────────────────────────────────────────────────────

def load_npy(file) -> np.ndarray:
    """Load MRNet-format numpy array."""
    return np.load(file)


def load_dicom(file) -> np.ndarray:
    """Load a single DICOM file or a folder of DICOM slices."""
    if not DICOM_AVAILABLE:
        st.error("pydicom is not installed. Cannot read DICOM files.")
        return None

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    ds     = pydicom.dcmread(tmp_path)
    volume = ds.pixel_array.astype(np.float32)
    os.unlink(tmp_path)

    # If single slice, add slice dimension
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]

    return volume


def load_nifti(file) -> np.ndarray:
    """Load NIfTI (.nii or .nii.gz) MRI file."""
    if not NIFTI_AVAILABLE:
        st.error("nibabel is not installed. Cannot read NIfTI files.")
        return None

    suffix = ".nii.gz" if file.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    img    = nib.load(tmp_path)
    volume = np.array(img.dataobj).astype(np.float32)
    os.unlink(tmp_path)

    # NIfTI is [H, W, slices] — transpose to [slices, H, W]
    if volume.ndim == 3:
        volume = np.transpose(volume, (2, 0, 1))

    return volume


def load_scan(uploaded_file) -> np.ndarray:
    """Route file to the correct loader based on extension."""
    name = uploaded_file.name.lower()

    if name.endswith(".npy"):
        return load_npy(uploaded_file)
    elif name.endswith(".dcm"):
        return load_dicom(uploaded_file)
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        return load_nifti(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload .npy, .dcm, or .nii / .nii.gz")
        return None


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_volume(volume: np.ndarray, device) -> torch.Tensor:
    """Normalize and resize MRI volume for model input."""
    # Normalize to [0, 1]
    v_min  = volume.min()
    v_max  = volume.max()
    volume = (volume - v_min) / (v_max - v_min + 1e-8)

    slices = []
    for s in range(volume.shape[0]):
        t = torch.tensor(volume[s], dtype=torch.float32).unsqueeze(0)
        t = TF.resize(t, [224, 224], antialias=True)
        t = t.repeat(3, 1, 1)   # grayscale → 3 channels
        slices.append(t)

    return torch.stack(slices).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

def predict(model, volume_tensor, threshold=0.5):
    """Run inference and return probability + decision."""
    with torch.no_grad():
        logit       = model(volume_tensor)
        probability = torch.sigmoid(logit).item()

    decision   = "ACL TEAR DETECTED" if probability >= threshold else "NO TEAR DETECTED"
    confidence = probability if probability >= threshold else 1 - probability
    return probability, decision, confidence


# ─────────────────────────────────────────────────────────────
# MRI VIEWER
# ─────────────────────────────────────────────────────────────

def render_mri_viewer(volume: np.ndarray):
    """Interactive MRI slice viewer with slider."""
    st.subheader("MRI Slice Viewer")

    num_slices  = volume.shape[0]
    slice_idx   = st.slider(
        "Scroll through slices",
        min_value   = 0,
        max_value   = num_slices - 1,
        value       = num_slices // 2,
        help        = "Move slider to inspect different slices of the MRI scan"
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(volume[slice_idx], cmap="gray")
    ax.set_title(f"Slice {slice_idx + 1} of {num_slices}", fontsize=11)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# RESULT CARD
# ─────────────────────────────────────────────────────────────

def render_result(probability, decision, confidence, threshold):
    """Render color-coded prediction result card."""
    tear_detected = probability >= threshold

    if tear_detected:
        st.error(f"### 🔴 {decision}")
        color_note = "The model detected signs consistent with an ACL tear."
    else:
        st.success(f"### 🟢 {decision}")
        color_note = "The model did not detect signs of an ACL tear."

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability",  f"{probability:.4f}")
    col2.metric("Confidence",   f"{confidence * 100:.1f}%")
    col3.metric("Threshold",    f"{threshold:.2f}")

    st.caption(color_note)

    st.divider()
    st.warning(
        "⚠️ **Clinical Disclaimer:** This tool is an AI-assisted decision "
        "support system and is not a substitute for professional medical "
        "diagnosis. All results must be reviewed and confirmed by a qualified "
        "radiologist or orthopaedic clinician before any clinical decision is made."
    )


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title = "ACL Tear Detection",
        page_icon  = "🦴",
        layout     = "centered"
    )

    # ── Header ────────────────────────────────────────────────
    st.title("🦴 ACL Tear Detection")
    st.markdown(
        "Upload a knee MRI scan to receive an automated assessment "
        "of ACL integrity. Supports sagittal plane MRI in **.npy**, "
        "**.dcm**, and **.nii / .nii.gz** formats."
    )
    st.divider()

    # ── Load model ────────────────────────────────────────────
    with st.spinner("Loading model..."):
        try:
            model, device = load_model()
            st.success(f"✓ Model loaded — running on **{str(device).upper()}**")
        except FileNotFoundError:
            st.error(
                "Model file `best_model.pt` not found. "
                "Make sure it is in the same directory as `app.py`."
            )
            st.stop()

    st.divider()

    # ── Sidebar settings ──────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            "Classification Threshold",
            min_value = 0.1,
            max_value = 0.9,
            value     = 0.5,
            step      = 0.05,
            help      = (
                "Probability cutoff for positive prediction. "
                "Lower = more sensitive (catches more tears, more false alarms). "
                "Higher = more specific (fewer false alarms, may miss some tears)."
            )
        )
        st.caption(
            "**Default: 0.5**  \n"
            "For clinical screening, consider lowering to 0.3 "
            "to maximize sensitivity."
        )

        st.divider()
        st.header("📋 Supported Formats")
        st.markdown(
            "- `.npy` — MRNet research format\n"
            "- `.dcm` — DICOM (hospital standard)\n"
            "- `.nii` / `.nii.gz` — NIfTI format"
        )
        st.divider()
        st.header("ℹ️ About")
        st.markdown(
            "This system uses a ResNet18-based deep learning model "
            "trained on the Stanford MRNet dataset to classify "
            "ACL integrity from sagittal knee MRI scans.\n\n"
            "**AUC-ROC: 0.9719**"
        )

    # ── File upload ───────────────────────────────────────────
    st.subheader("Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type       = ["npy", "dcm", "nii", "gz"],
        help       = "Upload a sagittal knee MRI scan"
    )

    if uploaded_file is not None:
        st.divider()

        # Load scan
        with st.spinner("Loading scan..."):
            volume = load_scan(uploaded_file)

        if volume is None:
            st.stop()

        st.success(
            f"✓ Scan loaded — **{volume.shape[0]} slices**, "
            f"resolution **{volume.shape[1]} × {volume.shape[2]}**"
        )

        # MRI viewer
        st.divider()
        render_mri_viewer(volume)

        # Run prediction
        st.divider()
        st.subheader("Prediction")

        with st.spinner("Analysing scan..."):
            volume_tensor              = preprocess_volume(volume, device)
            probability, decision, confidence = predict(model, volume_tensor, threshold)

        render_result(probability, decision, confidence, threshold)


if __name__ == "__main__":
    main()
