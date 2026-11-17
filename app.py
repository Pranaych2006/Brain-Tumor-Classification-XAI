
import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from model_utils import (
    load_trained_model,
    preprocess_image,
    integrated_gradients,
    occlusion_sensitivity,
    CLASS_NAMES,
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Brain Tumor Classification & XAI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM DASHBOARD DESIGN ----------------

st.markdown("""
<style>

    .stApp {
        background-color: #F5F7FB;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    h1 {
        color: #172554;
        font-weight: 750;
    }

    h2, h3 {
        color: #1E3A5F;
        font-weight: 650;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    [data-testid="stFileUploader"] {
        background-color: white;
        border: 1px dashed #94A3B8;
        padding: 15px;
        border-radius: 12px;
    }

    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 9px;
        border: none;
        padding: 10px 22px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
        border: none;
    }

    [data-testid="stTabs"] button {
        font-weight: 600;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #E2E8F0;
    }

</style>
""", unsafe_allow_html=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    APP_DIR,
    "best_hybrid_vit_gru.keras"
)


@st.cache_resource
def get_model():
    return load_trained_model(MODEL_PATH)


# ---------------- HEATMAP VISUALIZATION ----------------

def create_heatmap(original_image, attribution_map):

    heatmap = np.asarray(attribution_map, dtype=np.float32)
    heatmap = np.squeeze(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(
            f"Expected a 2D heatmap, received {heatmap.shape}"
        )

    heatmap = np.nan_to_num(heatmap)
    heatmap = np.clip(heatmap, 0, 1)

    grayscale = Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    )

    grayscale = grayscale.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    colored_heatmap = ImageOps.colorize(
        grayscale,
        black="#14213d",
        white="#ff3b30"
    )

    overlay = Image.blend(
        original_image.convert("RGB"),
        colored_heatmap,
        alpha=0.45
    )

    return colored_heatmap, overlay


# ---------------- PDF REPORT GENERATOR ----------------

def generate_pdf_report(
    original_image,
    predicted_class,
    confidence,
    predictions,
    ig_map,
    black_map,
    blur_map
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            "Brain Tumor Classification Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Hybrid Vision Transformer + GRU with Explainable AI",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            f"<b>Predicted Class:</b> {predicted_class}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Model Confidence:</b> {confidence:.2f}%",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph("Class Probability Results", styles["Heading2"])
    )

    probability_data = [
        ["Class", "Probability"]
    ]

    for name, probability in zip(CLASS_NAMES, predictions):
        probability_data.append([
            name,
            f"{float(probability) * 100:.2f}%"
        ])

    probability_table = Table(
        probability_data,
        colWidths=[220, 180]
    )

    probability_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(probability_table)

    elements.append(Spacer(1, 20))

    # Convert PIL images to PDF-compatible image objects
    def add_image_section(title, pil_image):

        if pil_image is None:
            return

        image_buffer = io.BytesIO()
        pil_image.convert("RGB").save(
            image_buffer,
            format="PNG"
        )
        image_buffer.seek(0)

        elements.append(
            Paragraph(title, styles["Heading2"])
        )

        elements.append(Spacer(1, 8))

        elements.append(
            PDFImage(
                image_buffer,
                width=3.8 * inch,
                height=3.8 * inch,
                kind="proportional"
            )
        )

        elements.append(Spacer(1, 15))

    # Original MRI
    add_image_section("Original MRI Image", original_image)

    # XAI visualizations
    if ig_map is not None:
        _, ig_overlay = create_heatmap(
            original_image,
            ig_map
        )
        add_image_section(
            "Integrated Gradients Overlay",
            ig_overlay
        )

    if black_map is not None:
        _, black_overlay = create_heatmap(
            original_image,
            black_map
        )
        add_image_section(
            "Black-patch Occlusion Overlay",
            black_overlay
        )

    if blur_map is not None:
        _, blur_overlay = create_heatmap(
            original_image,
            blur_map
        )
        add_image_section(
            "Blur-based Occlusion Overlay",
            blur_overlay
        )

    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            "<b>Research Disclaimer:</b> This report is generated "
            "by an academic research prototype. It is not clinically "
            "validated. Model predictions and XAI visualizations must "
            "not be used for medical decisions. Heatmaps indicate "
            "model attribution or sensitivity, not confirmed tumor boundaries.",
            styles["Normal"]
        )
    )

    document.build(elements)

    buffer.seek(0)

    return buffer.getvalue()



# ---------------- WEBSITE HEADER ----------------

st.title("🧠 Brain Tumor Classification & Explainable AI")

st.write(
    "An academic MRI classification application "
    "powered by Hybrid Vision Transformer + GRU."
)

st.info(
    "Upload a brain MRI image to obtain a classification "
    "and explore the model's explanations."
)

# ---------------- IMAGE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded MRI Image")
    st.image(image, width=400)

    if st.button("Analyze MRI Image", type="primary"):

        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found!")
            st.stop()

        try:
            model = get_model()
            processed_image = preprocess_image(image)

            # -------- PREDICTION --------

            with st.spinner("Classifying MRI image..."):

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

            st.session_state["prediction"] = predictions
            st.session_state["predicted_class"] = predicted_class
            st.session_state["confidence"] = confidence

            # -------- INTEGRATED GRADIENTS --------

            try:
                with st.spinner("Generating Integrated Gradients..."):

                    ig_map, _ = integrated_gradients(
                        model,
                        processed_image,
                        target_class=predicted_index,
                        steps=64
                    )

                st.session_state["ig_map"] = ig_map
                st.session_state["ig_error"] = None

            except Exception as e:
                st.session_state["ig_error"] = str(e)
                st.session_state["ig_map"] = None

            # -------- BLACK OCCLUSION --------

            try:
                with st.spinner("Generating Black-patch Occlusion..."):

                    black_map, _, _ = occlusion_sensitivity(
                        model,
                        processed_image,
                        target_class=predicted_index,
                        patch_size=16,
                        batch_size=32,
                        mode="black"
                    )

                st.session_state["black_map"] = black_map
                st.session_state["black_error"] = None

            except Exception as e:
                st.session_state["black_error"] = str(e)
                st.session_state["black_map"] = None

            # -------- BLUR OCCLUSION --------

            try:
                with st.spinner("Generating Blur-based Occlusion..."):

                    blur_map, _, _ = occlusion_sensitivity(
                        model,
                        processed_image,
                        target_class=predicted_index,
                        patch_size=16,
                        batch_size=32,
                        mode="blur"
                    )

                st.session_state["blur_map"] = blur_map
                st.session_state["blur_error"] = None

            except Exception as e:
                st.session_state["blur_error"] = str(e)
                st.session_state["blur_map"] = None

            st.session_state["analysis_image"] = image

            st.success("Analysis completed!")

        except Exception as e:
            st.error(f"Analysis failed: {e}")


# ---------------- PREDICTION RESULTS ----------------

if "prediction" in st.session_state:

    st.divider()
    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Class",
            st.session_state["predicted_class"]
        )

    with col2:
        st.metric(
            "Model Confidence",
            f'{st.session_state["confidence"]:.2f}%'
        )

    st.subheader("Class Probabilities")

    probabilities = st.session_state["prediction"]

    for name, probability in zip(CLASS_NAMES, probabilities):

        st.write(
            f"**{name}** — {probability * 100:.2f}%"
        )

        st.progress(float(probability))

    # ---------------- XAI SECTION ----------------

    st.divider()
    st.header("🔍 Explainable AI (XAI)")

    st.write(
        "Explore how different image regions influence "
        "the model's prediction."
    )

    image = st.session_state["analysis_image"]

    tabs = st.tabs([
        "Integrated Gradients",
        "Black-patch Occlusion",
        "Blur-based Occlusion"
    ])

    methods = [
        (
            "ig_map",
            "ig_error",
            "Integrated Gradients",
            "Highlights positive pixel-level attribution "
            "relative to a black baseline."
        ),
        (
            "black_map",
            "black_error",
            "Black-patch Occlusion Sensitivity",
            "Measures how replacing image patches with black "
            "regions changes the selected class score."
        ),
        (
            "blur_map",
            "blur_error",
            "Blur-based Occlusion Sensitivity",
            "Measures how blurring image patches changes "
            "the selected class score."
        )
    ]

    
# ---------------- ENHANCED XAI VISUALIZATION ----------------

    for tab, method in zip(tabs, methods):

        map_key, error_key, heading, description = method

        with tab:

            st.subheader(heading)

            st.info(description)

            heatmap = st.session_state.get(map_key)

            if heatmap is not None:

                colored, overlay = create_heatmap(
                    image,
                    heatmap
                )

                st.write("### Visual Comparison")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        image,
                        caption="Original MRI",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        colored,
                        caption="XAI Heatmap",
                        use_container_width=True
                    )

                with col3:
                    st.image(
                        overlay,
                        caption="MRI + Heatmap Overlay",
                        use_container_width=True
                    )

                st.divider()

                st.write("### How to interpret this visualization")

                st.markdown("""
                - **Original MRI:** The uploaded brain MRI image.
                - **XAI Heatmap:** Highlights regions receiving stronger
                attribution or sensitivity scores.
                - **Overlay:** Combines the original MRI with the heatmap
                to make the highlighted regions easier to inspect.

                Warmer colours indicate stronger values in the displayed
                normalized heatmap. Their exact meaning depends on the
                explanation method.
                """)

            else:

                st.error(
                    "This XAI visualization could not be generated."
                )

                error = st.session_state.get(error_key)

                if error:
                    st.code(error)


    # ---------------- DISCLAIMER ----------------

    st.warning(
        "Disclaimer: This is an academic research prototype, "
        "not a clinically validated diagnostic system. "
        "XAI heatmaps represent model attribution or sensitivity, "
        "not confirmed tumor boundaries. Do not use this application "
        "for medical decisions."
    )
    # ---------------- PDF DOWNLOAD ----------------

    st.divider()
    st.header("📄 Download Analysis Report")

    try:
        pdf_data = generate_pdf_report(
            original_image=st.session_state["analysis_image"],
            predicted_class=st.session_state["predicted_class"],
            confidence=st.session_state["confidence"],
            predictions=st.session_state["prediction"],
            ig_map=st.session_state.get("ig_map"),
            black_map=st.session_state.get("black_map"),
            blur_map=st.session_state.get("blur_map")
        )

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_data,
            file_name="Brain_Tumor_Analysis_Report.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"PDF generation failed: {e}")
