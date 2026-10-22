import numpy as np

def validate_scan(volume: np.ndarray):
    """
    Basic sanity checks on the uploaded scan.
    Returns (is_valid, message).

    NOTE: These checks catch obviously wrong files (wrong shape,
    wrong dimensions) but cannot guarantee the scan is a knee —
    a properly trained gatekeeper classifier would be needed for
    that. This is a lightweight first line of defense.
    """
    if volume is None:
        return False, "Could not read the uploaded file."

    if volume.ndim != 3:
        return False, (
            f"Expected a 3D MRI volume (slices, height, width), "
            f"but got {volume.ndim}D data with shape {volume.shape}. "
            f"Please upload a sagittal knee MRI scan."
        )

    if not np.issubdtype(volume.dtype, np.number):
        return False, "The scan must contain numeric voxel values."
    if volume.size == 0 or not np.isfinite(volume).all():
        return False, "The scan contains empty or non-finite voxel values."

    num_slices, height, width = volume.shape

    # Sagittal knee MRI scans in MRNet typically have 17-51 slices
    if not (10 <= num_slices <= 80):
        return False, (
            f"This scan has {num_slices} slices, which is outside the "
            f"expected range for a sagittal knee MRI (10-80 slices). "
            f"Please verify this is the correct scan type."
        )

    # MRNet images are 256x256, but allow some tolerance for other sources
    if not (150 <= height <= 512 and 150 <= width <= 512):
        return False, (
            f"This scan has dimensions {height}x{width}, which is outside "
            f"the expected range for a knee MRI. Please verify the uploaded file."
        )

    # Reject flat/empty volumes (all zeros or constant value)
    if volume.max() == volume.min():
        return False, "This file appears to contain no image data."

    return True, "Scan passed validation checks."

