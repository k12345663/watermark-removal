# Watermark Removal for Scanned Question Paper Images

## Problem Statement

We process thousands of scanned question paper images extracted from coaching institute PDFs (Narayana, Sri Chaitanya, etc.). These images contain **physics diagrams, chemistry structures, math figures, and biology illustrations** used in JEE/NEET exam preparation.

**The images have two problems:**

### 1. Colored Watermarks
Most images contain semi-transparent **blue, grey, or red watermark text/logos** overlaid on the question diagrams. These watermarks must be **completely removed** without disturbing the underlying diagram content (lines, labels, arrows, shading).

### 2. Low Contrast / Bluish Color Cast
Many scanned images have a **bluish color cast** (low color temperature) and **reduced contrast**, making text and diagram lines appear faded or washed out. The technical term is **desaturation with a cool-tone shift**. These images need **contrast enhancement and color correction** to produce sharp, high-contrast black-on-white output suitable for digital display.

### 3. Clean Image Detection
Not all images have watermarks. The solution must **automatically detect** whether an image needs processing and **skip clean images** (or pass them through unchanged) to avoid unnecessary quality degradation.

---

## What Success Looks Like

| Input | Output |
|-------|--------|
| Bluish, low-contrast image with visible watermark text behind a physics diagram | Sharp, high-contrast **grayscale** image with clean white background, crisp black lines, no trace of watermark |
| Clean image with no watermark | Passed through unchanged (or with minimal enhancement) |
| Image with red "TG" or "bohring bot" watermark | Watermark completely removed, diagram intact |

**Key constraint:** The diagram content (lines, arrows, text labels, hatching, shading) must be **preserved exactly**. Aggressive thresholding that destroys fine detail is not acceptable.

---

## Sample Images

The `samples/` directory contains **150 real images** from our pipeline:

- **`samples/watermarked/`** (120 images) — Images with visible blue/grey watermarks at varying intensities. These are the primary challenge.
- **`samples/clean/`** (30 images) — Images without watermarks. Your solution should detect these and skip/pass-through.

Sources: Narayana WAT Collection, Narayana DPPs, Sri Chaitanya GTMS 2025 (Physics, Chemistry, Mathematics, Biology).

---

## Current Approach (Baseline)

The current solution is in `src/remove_watermark.py`. It uses a **histogram-based LUT approach**:

1. **Red watermark pre-check** — HSV color masking detects red pixels (H near 0/180, S > 50), removes them with dilation to catch anti-aliased edges
2. **Convert to grayscale** — luminance-based conversion preserves black text
3. **Histogram analysis** — find the brightness boundary between text pixels (dark) and watermark pixels (light)
4. **LUT mapping** — smooth curve pushes light pixels (watermark + background) to pure white, darkens text pixels for contrast
5. **Auto-contrast stretch** — 2nd/98th percentile normalization
6. **Sharpening** — 3x3 unsharp mask kernel to preserve text edges
7. **Red mask re-application** — force formerly-red areas back to white (prevents dark outlines from LUT/sharpening)

### Running the baseline

```bash
pip install -r requirements.txt

# Process all watermarked samples
python src/remove_watermark.py samples/watermarked output/

# Process a single directory
python src/remove_watermark.py <input_dir> <output_dir>
```

Output appears in the `output/` directory as PNG files.

### Limitations of the current approach

- **Bluish tint not fully corrected** — the LUT approach converts to grayscale, losing potential color information that could help distinguish watermark from diagram
- **Fixed threshold range** — the `[160, 210]` adaptive range doesn't handle all watermark intensities well
- **No per-image classification** — doesn't detect whether an image actually needs processing
- **Sharpening artifacts** — the 3x3 kernel can create halos around high-contrast edges
- **No batch quality metrics** — no automated way to measure output quality

---

## Suggested Improvements

You are free to explore **any approach**. Here are some directions to consider:

### Approach A: Classical CV (improve current)
- Better colorspace separation (LAB, YCbCr) to isolate watermark from content
- Adaptive per-image thresholding based on histogram shape (bimodal detection)
- Morphological operations to preserve fine lines during watermark removal
- Edge-aware filtering (bilateral filter, guided filter) instead of basic sharpening
- Clean-image classifier based on histogram statistics (skip processing if unnecessary)

### Approach B: Frequency Domain
- FFT/DCT analysis — watermarks often have periodic patterns that can be filtered in frequency space
- Wavelet decomposition — separate watermark texture from diagram structure

### Approach C: Deep Learning
- U-Net or similar encoder-decoder for image restoration
- Pre-trained denoising models (e.g., DnCNN, NAFNet, Restormer)
- GAN-based inpainting for watermark regions
- Train on paired data (watermarked vs clean versions of same diagram)

### Approach D: Color-Channel Analysis
- Since watermarks are typically a single color (blue), isolate that channel and subtract
- LAB colorspace: watermark affects `a` and `b` channels, content is primarily in `L`
- Independent component analysis (ICA) to separate watermark signal from content

---

## Evaluation Criteria

Your solution will be evaluated on:

1. **Watermark removal quality** — Is the watermark fully gone? No ghost traces?
2. **Content preservation** — Are diagram lines, arrows, labels, and fine details intact?
3. **Clean image handling** — Does it correctly detect and skip clean images?
4. **Contrast/sharpness** — Is the output sharp and high-contrast (black on white)?
5. **Robustness** — Does it handle varying watermark colors, intensities, and positions?
6. **Speed** — Reasonable processing time per image (< 2 seconds per image is fine)

---

## Tech Stack

**Required:** Python 3.10+

**Current dependencies:**
- OpenCV (`opencv-python`) — image I/O, color conversion, filtering
- NumPy — array operations, histogram analysis
- Pillow — fallback image processing

**You may add:**
- scikit-image — advanced morphology, segmentation
- PyTorch / TensorFlow — if using deep learning approaches
- scipy — frequency domain processing
- Any other Python library that helps

---

## Directory Structure

```
watermark-removal/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore
├── src/
│   └── remove_watermark.py   # Current baseline implementation
├── samples/
│   ├── watermarked/       # 120 images WITH watermarks (the challenge)
│   └── clean/             # 30 images WITHOUT watermarks (for skip detection)
└── output/                # Your processed output goes here (gitignored)
```

---

## Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd watermark-removal

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the baseline to see current output
python src/remove_watermark.py samples/watermarked output/

# Compare input vs output side by side
# (open samples/watermarked/wm_001.jpg and output/wm_001.png together)
```

---

## Deliverables

1. Updated `src/remove_watermark.py` (or new files in `src/`) with your improved approach
2. `output/` directory with processed results on all 120 watermarked samples
3. Brief write-up of your approach, what you tried, and results (can be in this README or a separate doc)
4. Any additional requirements added to `requirements.txt`
