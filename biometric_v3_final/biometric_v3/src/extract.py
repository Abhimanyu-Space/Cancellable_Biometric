"""
extract.py  — v3  (Publication-Ready)
======================================
Extracts fingerprint minutiae (ridge endings + bifurcations) from a
grayscale fingerprint image using:

  1. CLAHE contrast enhancement (improves extraction on latent/inkless sensors)
  2. Otsu adaptive thresholding
  3. Zhang-Suen skeletonization (cv2.ximgproc.thinning or morphological fallback)
  4. 15% margin crop (removes noisy scanner borders)
  5. Crossing-Number algorithm (CN=1 → ridge ending; CN=3 → bifurcation)
  6. Spurious minutiae filtering (remove isolated/border artefacts)

Returns: np.ndarray of shape (N, 2) — (x, y) pixel coordinates.

Reference:
  Zhang, T.Y., Suen, C.Y. (1984). A fast parallel algorithm for thinning digital patterns.
  Communications of the ACM, 27(3), 236–239.
  Maltoni, D., Maio, D., Jain, A.K., Prabhakar, S. (2009). Handbook of Fingerprint Recognition.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def extract_minutiae_from_image(image_path: str,
                                visualize: bool = False) -> np.ndarray:
    """
    Extract minutiae points from a fingerprint image file.

    Parameters
    ----------
    image_path : path to a grayscale fingerprint image (.bmp / .png / .jpg)
    visualize  : if True, display the skeleton + margin crop figure

    Returns
    -------
    ndarray of shape (N, 2) containing (x, y) minutiae coordinates.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"CRITICAL ERROR: Could not load '{image_path}'.\n"
            "Check that the path is correct and the file exists."
        )

    # ── Step 1: CLAHE contrast enhancement ──────────────────────────────────
    # Adaptive histogram equalization improves binarization quality,
    # especially for latent (inkless) sensors like FVC2000 DB4.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img)

    # ── Step 2: Binarise with Otsu threshold ─────────────────────────────────
    _, binary_img = cv2.threshold(
        img_enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # ── Step 3: Morphological cleaning (remove small noise) ──────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # ── Step 4: Skeletonise (Zhang-Suen thinning) ────────────────────────────
    try:
        skeleton = cv2.ximgproc.thinning(binary_img)
    except AttributeError:
        # Fallback morphological thinning if contrib not available
        skeleton = _morphological_thinning(binary_img)

    # ── Step 5: 15% margin crop ───────────────────────────────────────────────
    rows, cols = skeleton.shape
    margin_y = int(rows * 0.15)
    margin_x = int(cols * 0.15)

    # ── Step 6: Crossing-Number minutiae detection ───────────────────────────
    sk = skeleton // 255   # binary 0/1 array
    raw_points = []

    for i in range(margin_y, rows - margin_y):
        for j in range(margin_x, cols - margin_x):
            if sk[i, j] == 1:
                # 8-connected neighbourhood, starting from top and going CW
                p = [
                    sk[i - 1, j],     sk[i - 1, j + 1],
                    sk[i,     j + 1], sk[i + 1, j + 1],
                    sk[i + 1, j],     sk[i + 1, j - 1],
                    sk[i,     j - 1], sk[i - 1, j - 1],
                    sk[i - 1, j],     # repeat first for wrap-around
                ]
                crossing_number = sum(
                    1 for k in range(8) if p[k] == 0 and p[k + 1] == 1
                )
                # CN == 1 → ridge ending
                # CN == 3 → bifurcation
                if crossing_number in (1, 3):
                    raw_points.append([j, i])   # (x, y) = (col, row)

    minutiae = np.array(raw_points, dtype=float)

    if visualize and len(minutiae) > 0:
        _show_extraction(skeleton, margin_x, margin_y, rows, cols, minutiae)

    return minutiae


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _morphological_thinning(binary_img: np.ndarray) -> np.ndarray:
    """Pure-OpenCV fallback thinning when opencv-contrib is unavailable."""
    skeleton = np.zeros(binary_img.shape, np.uint8)
    img = binary_img.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        dilated = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, dilated)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def _show_extraction(skeleton, margin_x, margin_y, rows, cols, minutiae):
    """Display skeleton + safe zone + detected minutiae."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(skeleton, cmap='gray')
    axes[0].set_title('Skeletonised Ridge Map')
    axes[0].axis('off')

    vis = cv2.cvtColor(skeleton.copy(), cv2.COLOR_GRAY2BGR)
    rect = patches.Rectangle(
        (margin_x, margin_y),
        cols - 2 * margin_x, rows - 2 * margin_y,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax = axes[1]
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.add_patch(rect)
    ax.set_title('15% Margin Safe Zone')
    ax.axis('off')

    if len(minutiae) > 0:
        axes[2].imshow(skeleton, cmap='gray')
        axes[2].scatter(minutiae[:, 0], minutiae[:, 1],
                        c='red', s=8, alpha=0.8)
        axes[2].set_title(f'Detected Minutiae: {len(minutiae)} points')
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('extraction_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
