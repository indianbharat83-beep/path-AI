import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io

st.set_page_config(page_title='PathAI Demo MVP', layout='centered')

st.title('Pathology Diagnostic Support — Demo MVP')
st.write('Upload a pathology slide image. The app highlights suspicious regions (demo heuristic) and generates a draft report.')

uploaded = st.file_uploader('Upload a pathology slide image (PNG/JPG)', type=['png', 'jpg', 'jpeg'])

def analyze_image_no_matplotlib(pil_img):
    # Convert to grayscale numpy array
    gray = ImageOps.grayscale(pil_img)
    arr = np.array(gray).astype(float)

    # Heuristic: pixels brighter than mean + 0.5*std flagged as suspicious
    mean = arr.mean()
    std = arr.std()
    thresh = mean + 0.5 * std
    mask = (arr > thresh).astype(np.uint8)  # 0 or 1 mask

    # Create overlay image (red tint) from mask with soft alpha
    # Normalize mask to 0..255 alpha
    alpha = (mask * 180).astype(np.uint8)  # 0..180 alpha for overlay
    # Create RGBA overlay: red color where mask==1
    overlay = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 255          # Red channel
    overlay[..., 3] = alpha       # Alpha channel

    # Convert base grayscale to RGB
    base_rgb = np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
    base_img = Image.fromarray(base_rgb, mode='RGB')

    overlay_img = Image.fromarray(overlay, mode='RGBA')

    # Composite overlay on base image
    composed = Image.alpha_composite(base_img.convert('RGBA'), overlay_img).convert('RGB')

    # Also produce a simple binary mask image for preview
    mask_img = (mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img).convert('RGB')

    return mask, composed, mean, std, mask_pil

def generate_report(mean, std, mask):
    suspicious_pct = (mask > 0).mean() * 100
    report = f"""Draft Report (automatically generated):

Findings:
- Mean tissue intensity: {mean:.2f}
- Intensity variability (std): {std:.2f}
- Suspicious regions detected covering ~{suspicious_pct:.2f}% of the slide area.

Impression:
- Areas flagged may indicate regions of interest requiring pathologist review.
- Recommend targeted review and, if needed, additional stains or higher-power examination.

Notes:
- This is a demo. Final diagnosis must be confirmed by a qualified pathologist.
"""
    return report

if uploaded:
    try:
        pil_img = Image.open(uploaded).convert('RGB')
    except Exception as e:
        st.error(f"Unable to open uploaded file: {e}")
        st.stop()

    st.subheader('Original Image')
    st.image(pil_img, use_column_width=True)

    with st.spinner('AI analyzing... (demo heuristic)'):
        mask, heatmap, mean, std, mask_preview = analyze_image_no_matplotlib(pil_img)

    st.subheader('AI Heatmap Overlay (demo)')
    st.image(heatmap, use_column_width=True)

    st.subheader('Detected Mask Preview')
    st.image(mask_preview, use_column_width=True)

    report = generate_report(mean, std, mask)
    st.subheader('Draft Report')
    st.code(report)

    st.download_button('Download Draft Report', report, file_name='draft_report.txt')
