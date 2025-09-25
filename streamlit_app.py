
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import io

st.set_page_config(page_title='PathAI Demo MVP', layout='centered')

st.title('Pathology Diagnostic Support — Demo MVP')
st.write('Functional demo: uploads an image, highlights suspicious regions (simple heuristic), and generates a draft report.')
uploaded = st.file_uploader('Upload a pathology slide image (PNG/JPG)', type=['png','jpg','jpeg'])

def analyze_image(pil_img):
    # Convert to grayscale and normalize
    gray = ImageOps.grayscale(pil_img)
    arr = np.array(gray).astype(float)
    # Simple heuristic: areas brighter than mean + std considered suspicious
    mean = arr.mean()
    std = arr.std()
    thresh = mean + 0.5*std
    mask = (arr > thresh).astype(np.uint8) * 255
    # Create heatmap
    import matplotlib.pyplot as plt
    plt.imshow(arr, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.4)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    heatmap = Image.open(buf).convert('RGB')
    return mask, heatmap, mean, std

def generate_report(mean, std, mask):
    suspicious_pct = (mask>0).mean() * 100
    report = f"""Draft Report (automatically generated):

Findings:
- Mean tissue intensity: {mean:.2f}
- Intensity variability (std): {std:.2f}
- Suspicious regions detected covering approximately {suspicious_pct:.2f}% of the slide area.

Impression:
- Areas flagged may indicate regions of interest requiring pathologist review.
- Recommend targeted review and, if needed, additional stains or higher-power examination.

Notes:
- This is a demo. Final diagnosis must be confirmed by a qualified pathologist."""
    return report

if uploaded:
    pil_img = Image.open(uploaded).convert('RGB')
    st.subheader('Original Image')
    st.image(pil_img, use_column_width=True)
    with st.spinner('AI analyzing... (demo heuristic)'):
        mask, heatmap, mean, std = analyze_image(pil_img)
    st.subheader('AI Heatmap Overlay (demo)')
    st.image(heatmap, use_column_width=True)
    st.subheader('Detected Mask Preview')
    st.image(Image.fromarray(mask).convert('RGB'), use_column_width=True)
    report = generate_report(mean, std, mask)
    st.subheader('Draft Report')
    st.code(report)
    st.download_button('Download Draft Report', report, file_name='draft_report.txt')    
