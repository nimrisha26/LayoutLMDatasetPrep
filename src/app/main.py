import streamlit as st
import pandas as pd
import os
import sys
import tempfile
import shutil
import uuid
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# ---- Import your project modules ----
sys.path.append("C:/Users/pavit/LayoutLM")
from src.main import process_image
from src.parsing.pdf2img import pdf_to_images
from src.labelling.highlight_labels import highlight_labels


# ---- Streamlit Setup ----
st.set_page_config(page_title="LayoutLM Dataset Preparation", layout="wide")
st.title("📘 LayoutLM Dataset Preparation Tool")
st.caption("Upload PDF → Extract raw LayoutLM chunks → Visualize highlights → Download CSV")

# ---- State ----
if "df" not in st.session_state:
    st.session_state.df = None
if "highlight_images" not in st.session_state:
    st.session_state.highlight_images = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "out_dir" not in st.session_state:
    st.session_state.out_dir = None


# ---- Cached helper functions ----
@st.cache_data(show_spinner=False)
def cached_pdf_to_images(pdf_path):
    return pdf_to_images(pdf_path=pdf_path, IMAGES_DIR="Images")


@st.cache_data(show_spinner=False)
def cached_process_image(image_path, page_id):
    return process_image(image_path=image_path, page_id=page_id, doc_id=0, ocr="doctr")


def process_and_highlight(pdf_path: str):
    """Run OCR + tokenization + highlighting pipeline."""
    out_dir = os.path.join("highlighted_images_output", f"run_{uuid.uuid4().hex[:8]}")
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: PDF → images
    img_paths = cached_pdf_to_images(pdf_path)

    # Step 2: Process pages (multi-threaded)
    results_all = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cached_process_image, p, i) for i, p in enumerate(img_paths)]
        for i, f in enumerate(futures):
            page_results = f.result()  # list of chunk dicts
            results_all.extend(page_results)  # flatten directly
            highlight_labels(page_results, out_dir)
            st.progress((i + 1) / len(img_paths))

    # Step 3: Convert to DataFrame directly
    df = pd.DataFrame(results_all)

    # Step 4: Collect highlighted images
    highlight_images = [
        os.path.join(out_dir, f)
        for f in sorted(os.listdir(out_dir))
        if f.lower().endswith(".png")
    ]

    return df, highlight_images, out_dir


# ---- PDF Upload ----
uploaded_pdf = st.file_uploader("📂 Upload PDF file", type=["pdf"])

if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_pdf.read())
        pdf_path = tmp_pdf.name

    st.success("✅ PDF uploaded successfully!")

    # ---- Run Processing ----
    if st.button("🚀 Run OCR & Prepare Dataset"):
        with st.spinner("🔄 Processing PDF..."):
            # Clean previous run
            if st.session_state.out_dir and os.path.exists(st.session_state.out_dir):
                shutil.rmtree(st.session_state.out_dir, ignore_errors=True)

            df, highlight_images, out_dir = process_and_highlight(pdf_path)

            st.session_state.df = df
            st.session_state.highlight_images = highlight_images
            st.session_state.out_dir = out_dir
            st.session_state.processed = True

        st.success("✅ Processing completed successfully!")


# ---- Display Results ----
if st.session_state.processed:
    st.subheader("📊 Extracted Raw Chunks Data")
    st.dataframe(st.session_state.df, use_container_width=True, height=400)

    # ---- Download CSV ----
    csv_data = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Dataset as CSV",
        data=csv_data,
        file_name="layoutlm_chunks_raw.csv",
        mime="text/csv",
    )

    # ---- Highlight Viewer ----
    st.subheader("🖼️ Highlighted Pages Viewer")

    if st.session_state.highlight_images:
        options = [f"Page {i+1}" for i in range(len(st.session_state.highlight_images))]
        selected_page = st.selectbox("Select a page to view:", options, index=0)

        page_index = options.index(selected_page)
        img_path = st.session_state.highlight_images[page_index]
        img = Image.open(img_path)

        # Resize image for better Streamlit fit
        img.thumbnail((1200, 1600))
        st.image(img, caption=selected_page, use_container_width=False)
    else:
        st.warning("⚠️ No highlighted images generated.")
else:
    st.info("⬆️ Upload a PDF and click **Run OCR & Prepare Dataset** to start.")
