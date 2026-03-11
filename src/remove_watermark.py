"""Watermark removal for scanned question paper images.

Removes colored watermarks (blue, red, grey) while preserving black text,
diagrams, and line art. Converts to clean grayscale output.

Approach: Multi-method grayscale conversion + adaptive thresholding
- Step 0: Red watermark pre-removal (HSV color masking + dilation)
- Step 1: Grayscale via luminance (preserves black text)
- Step 2: Histogram analysis to find watermark/background boundary
- Step 3: LUT-based push: light pixels (watermark + bg) -> white
- Step 4: Auto-contrast stretch for remaining dark pixels (text/lines)
- Step 5: Gentle sharpening to preserve detail
- Step 6: Re-apply red mask to force formerly-red areas to white

Tech stack: OpenCV + NumPy (primary), Pillow (fallback)
"""

import io
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── Red watermark detection + removal ─────────────────────────────────────

def _remove_red_watermark(img_path: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Detect and remove red/dark-red watermark pixels.

    Uses HSV color space to target saturated red pixels (e.g., "bohring bot",
    "TG" labels) while preserving black text (low saturation) and diagram lines.

    Returns (temp_path, dilated_mask) or (None, None).
    The dilated mask is used AFTER grayscale processing to force those pixels
    back to white, preventing the sharpening/LUT from creating dark outlines.
    """
    if HAS_CV2:
        img = cv2.imread(img_path)
        if img is None:
            return None, None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red in HSV wraps around 0/180: H in [0,10] U [170,180]
        # S > 50 (must be saturated -- not grey/black text)
        # V > 50 (must have brightness -- not dark text)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Also catch dark maroon/brown-red: H in [0,15], S > 40, V > 30
        lower_maroon = np.array([0, 40, 30])
        upper_maroon = np.array([15, 255, 200])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask3 = cv2.inRange(hsv, lower_maroon, upper_maroon)
        red_mask = mask1 | mask2 | mask3

        red_pixel_count = np.count_nonzero(red_mask)
        total_pixels = img.shape[0] * img.shape[1]

        # Only apply if red pixels are < 15% of image (watermark, not a red diagram)
        if red_pixel_count < 10 or red_pixel_count > total_pixels * 0.15:
            return None, None

        # Dilate the mask by 3px to catch anti-aliased edge pixels around red text
        dilate_kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(red_mask, dilate_kernel, iterations=3)

        # Replace red pixels (and their edges) with white
        img[dilated_mask > 0] = [255, 255, 255]

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, img)
        return tmp.name, dilated_mask

    elif HAS_PIL:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return None, None

        arr = np.array(img, dtype=np.float32)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        red_mask = (r > 120) & (r > g * 1.5) & (r > b * 1.5)
        red_pixel_count = np.count_nonzero(red_mask)
        total_pixels = arr.shape[0] * arr.shape[1]

        if red_pixel_count < 10 or red_pixel_count > total_pixels * 0.15:
            return None, None

        # Manual dilation for PIL path
        dilated = red_mask.copy()
        for _ in range(3):
            padded = np.pad(dilated, 1, mode='constant', constant_values=False)
            dilated = (padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
                       padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] |
                       padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:])

        result = np.array(img)
        result[dilated] = [255, 255, 255]

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(result).save(tmp.name)
        return tmp.name, (dilated.astype(np.uint8) * 255)

    return None, None


# ── Core watermark removal ────────────────────────────────────────────────

def remove_watermark(img_path: str) -> Optional[bytes]:
    """Remove watermark from an image. Returns cleaned image as PNG bytes.

    This works because:
    - Text/diagrams are dark (grayscale < 128)
    - Watermarks are light colored or grey (grayscale > 160)
    - Background is white/near-white (grayscale > 230)
    """
    temp_path, red_mask = _remove_red_watermark(img_path)
    effective_path = temp_path if temp_path else img_path

    try:
        if HAS_CV2:
            return _remove_watermark_cv2(effective_path, red_mask=red_mask)
        elif HAS_PIL:
            return _remove_watermark_pil(effective_path, red_mask=red_mask)
        else:
            print("ERROR: Neither OpenCV nor Pillow installed.")
            return None
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _remove_watermark_cv2(img_path: str, red_mask=None) -> Optional[bytes]:
    """OpenCV-based watermark removal."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Analyze histogram to find watermark threshold
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    p5 = np.percentile(gray, 5)
    p50 = np.percentile(gray, 50)
    p90 = np.percentile(gray, 90)

    # Adaptive threshold: midpoint between text cluster and background
    threshold = int(min(max(p50 + 20, 160), 210))

    # LUT: push watermark pixels to white, enhance dark pixels (text)
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < threshold - 40:
            ratio = i / max(threshold - 40, 1)
            lut[i] = int(ratio * min(i, 180))
        elif i < threshold:
            ratio = (i - (threshold - 40)) / 40.0
            dark_val = int(min(i, 180))
            lut[i] = int(dark_val * (1 - ratio) + 255 * ratio)
        else:
            lut[i] = 255

    cleaned = cv2.LUT(gray, lut)

    # Auto-contrast stretch
    p_low = np.percentile(cleaned[cleaned < 250], 2) if np.any(cleaned < 250) else 0
    p_high = 255
    if p_low < p_high:
        cleaned = np.clip((cleaned.astype(float) - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)

    # Sharpening to preserve text edges
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    cleaned = cv2.filter2D(cleaned, -1, kernel)
    cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

    # Re-apply red mask
    if red_mask is not None and red_mask.shape == cleaned.shape:
        cleaned[red_mask > 0] = 255

    success, buffer = cv2.imencode('.png', cleaned)
    return buffer.tobytes() if success else None


def _remove_watermark_pil(img_path: str, red_mask=None) -> Optional[bytes]:
    """Pillow-based watermark removal (fallback)."""
    try:
        img = Image.open(img_path)
    except Exception:
        return None

    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    p50 = np.percentile(arr, 50)
    threshold = min(max(p50 + 20, 160), 210)

    result = np.where(arr > threshold, 255,
                      np.where(arr > threshold - 40,
                               255 * (arr - (threshold - 40)) / 40.0 + arr * (1 - (arr - (threshold - 40)) / 40.0),
                               arr))
    result = np.clip(result, 0, 255).astype(np.uint8)

    low = np.percentile(result[result < 250], 2) if np.any(result < 250) else 0
    if low < 255:
        result = np.clip((result.astype(float) - low) / (255 - low) * 255, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(result)
    out_img = out_img.filter(ImageFilter.SHARPEN)

    if red_mask is not None:
        result_final = np.array(out_img)
        if red_mask.shape == result_final.shape:
            result_final[red_mask > 0] = 255
            out_img = Image.fromarray(result_final)

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    return buf.getvalue()


# ── CLI entry point ───────────────────────────────────────────────────────

def process_directory(input_dir: str, output_dir: str):
    """Process all images in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    images = [f for f in sorted(input_path.iterdir()) if f.suffix.lower() in exts]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(images)} images...")
    processed = 0
    failed = 0

    for i, img_path in enumerate(images):
        cleaned_bytes = remove_watermark(str(img_path))
        out_file = output_path / img_path.with_suffix(".png").name

        if cleaned_bytes:
            with open(out_file, "wb") as f:
                f.write(cleaned_bytes)
            processed += 1
        else:
            import shutil
            shutil.copy2(img_path, output_path / img_path.name)
            failed += 1
            print(f"  FAILED: {img_path.name} (copied original)")

        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            print(f"  Progress: {i + 1}/{len(images)} ({(i+1)/len(images)*100:.0f}%)")

    print(f"\nDone: {processed} cleaned, {failed} failed")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_watermark.py <input_dir> [output_dir]")
        print("       python remove_watermark.py samples/watermarked output/")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    process_directory(input_dir, output_dir)
