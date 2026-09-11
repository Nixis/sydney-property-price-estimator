
# ============================================================
# SYDNEY PROPERTY PRICE ESTIMATOR
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sydney Property Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM IMPUTER REQUIRED BY THE SAVED PIPELINE
# ============================================================

class PropertyTypeNumericImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = X.copy()

        self.columns_ = [
            "land_size_m2",
            "parking_spaces"
        ]

        self.group_medians_ = {
            column: X.groupby("property_type")[column].median().to_dict()
            for column in self.columns_
        }

        self.overall_medians_ = (
            X[self.columns_].median().to_dict()
        )

        return self

    def transform(self, X):
        X = X.copy()

        for column in self.columns_:

            X[f"{column}_missing"] = (
                X[column].isna().astype(int)
            )

            matched_medians = X["property_type"].map(
                self.group_medians_[column]
            )

            X[column] = (
                X[column]
                .fillna(matched_medians)
                .fillna(self.overall_medians_[column])
            )

        return X


# ============================================================
# CUSTOM APPLICATION STYLING
# ============================================================

st.markdown(
    """
<style>

/* ------------------------------------------------------------
   MAIN PAGE
------------------------------------------------------------ */

.stApp {
    background: #ffffff;
    color: #0b2d63;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.3rem;
    padding-bottom: 2.5rem;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}


/* ------------------------------------------------------------
   HERO BANNER
------------------------------------------------------------ */

.hero-banner {
    position: relative;
    overflow: hidden;
    padding: 25px 32px;
    margin-bottom: 15px;
    border: 1px solid #dbe5ef;
    border-radius: 12px;
    background:
        linear-gradient(
            110deg,
            #ffffff 0%,
            #f6fbff 58%,
            #eaf6fd 100%
        );
}

.hero-banner::after {
    content: "⌂";
    position: absolute;
    right: 38px;
    top: -17px;
    color: rgba(255, 101, 0, 0.08);
    font-size: 145px;
    font-weight: 800;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-title {
    margin: 0;
    color: #082a5e;
    font-size: 2.35rem;
    font-weight: 850;
    line-height: 1.15;
}

.hero-subtitle {
    max-width: 850px;
    margin: 8px 0 15px 0;
    color: #496487;
    font-size: 1rem;
    line-height: 1.5;
}

.hero-badge {
    display: inline-block;
    padding: 6px 13px;
    margin-right: 7px;
    margin-bottom: 3px;
    border: 1px solid #cddcea;
    border-radius: 999px;
    background: #ffffff;
    color: #173c70;
    font-size: 0.84rem;
    font-weight: 650;
}


/* ------------------------------------------------------------
   ABOUT PROJECT
------------------------------------------------------------ */

.about-card {
    display: flex;
    align-items: flex-start;
    gap: 17px;
    padding: 18px 23px;
    margin-bottom: 14px;
    border: 1px solid #dce7f0;
    border-radius: 10px;
    background: #edf7fd;
}

.about-icon {
    flex-shrink: 0;
    color: #ff6500;
    font-size: 2rem;
    line-height: 1;
}

.about-title {
    margin-bottom: 5px;
    color: #082a5e;
    font-size: 1.15rem;
    font-weight: 800;
}

.about-text {
    margin: 0;
    color: #38577e;
    font-size: 0.95rem;
    line-height: 1.55;
}


/* ------------------------------------------------------------
   FORM CARDS
------------------------------------------------------------ */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #ccd8e5;
    border-radius: 9px;
    background: #ffffff;
    box-shadow: none;
}

.section-heading {
    display: flex;
    align-items: center;
    gap: 11px;
    min-height: 38px;
    margin: 0;
    color: #082a5e;
    font-size: 1.22rem;
    font-weight: 800;
    line-height: 1.2;
}

.section-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 31px;
    height: 31px;
    flex: 0 0 31px;
    color: #ff6500;
    font-size: 1.9rem;
    font-weight: 500;
    line-height: 1;
}

.section-description {
    margin: -2px 0 5px 42px;
    color: #657d9b;
    font-size: 0.88rem;
}


/* ------------------------------------------------------------
   INPUTS
------------------------------------------------------------ */

div[data-testid="stWidgetLabel"] p {
    color: #12376b;
    font-weight: 600;
}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInputContainer"] {
    border-radius: 6px;
    background: #ffffff;
}

div[data-baseweb="select"] > div:focus-within,
div[data-testid="stNumberInputContainer"]:focus-within {
    border-color: #ff6500;
}

input[type="checkbox"] {
    accent-color: #ff6500;
}


/* ------------------------------------------------------------
   ESTIMATE BUTTON
------------------------------------------------------------ */

div[data-testid="stFormSubmitButton"] button {
    min-height: 53px;
    border: 1px solid #ff6500;
    border-radius: 7px;
    background: #ff6500;
    color: #ffffff;
    font-size: 1.04rem;
    font-weight: 750;
    box-shadow: none;
}

div[data-testid="stFormSubmitButton"] button:hover {
    border-color: #e75700;
    background: #e75700;
    color: #ffffff;
}

div[data-testid="stFormSubmitButton"] button:focus {
    border-color: #e75700;
    box-shadow: 0 0 0 0.2rem rgba(255, 101, 0, 0.18);
}


/* ------------------------------------------------------------
   PREDICTION RESULT
------------------------------------------------------------ */

.prediction-card {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 20px 25px;
    margin: 13px 0;
    border: 1px solid #d5e5ef;
    border-radius: 10px;
    background:
        linear-gradient(
            110deg,
            #edf7fd 0%,
            #f8fcff 100%
        );
}

.prediction-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 78px;
    height: 64px;
    padding-right: 20px;
    flex-shrink: 0;
    border-right: 1px solid #bfcfdf;
}

.prediction-icon svg {
    width: 45px;
    height: 45px;
    stroke: #ff6500;
    stroke-width: 3.5;
}

.prediction-label {
    color: #163a6e;
    font-size: 0.95rem;
    font-weight: 700;
}

.prediction-value {
    margin: 1px 0;
    color: #082a5e;
    font-size: 2.05rem;
    font-weight: 850;
    line-height: 1.12;
}

.prediction-caption {
    color: #617895;
    font-size: 0.85rem;
}


/* ------------------------------------------------------------
   SUBMITTED INFORMATION
------------------------------------------------------------ */

div[data-testid="stExpander"] {
    border: 1px solid #d2dde8;
    border-radius: 8px;
    background: #ffffff;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #dce5ed;
    border-radius: 6px;
}


/* ------------------------------------------------------------
   SIDEBAR
------------------------------------------------------------ */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #eef8fd 0%,
            #f8fcff 100%
        );
    border-right: 1px solid #d7e4ee;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 5px 2px 21px 2px;
    margin-bottom: 8px;
    border-bottom: 1px solid #cad9e6;
}

.sidebar-brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 47px;
    height: 47px;
    flex: 0 0 47px;
    color: #ff6500;
    font-size: 2.8rem;
    font-weight: 500;
    line-height: 1;
    transform: translateY(-1px);
}

.sidebar-brand-name {
    margin: 0;
    color: #082a5e;
    font-size: 1.08rem;
    font-weight: 850;
    line-height: 1.23;
}

.sidebar-title {
    margin: 17px 0 13px 0;
    color: #082a5e;
    font-size: 1.05rem;
    font-weight: 800;
}

.sidebar-label {
    margin-top: 13px;
    color: #173c70;
    font-size: 0.82rem;
    font-weight: 750;
}

.sidebar-value {
    margin-top: 2px;
    color: #4e6888;
    font-size: 0.89rem;
    line-height: 1.4;
}

.limitation-card {
    padding: 14px;
    margin-top: 22px;
    border-left: 4px solid #ff6500;
    border-radius: 7px;
    background: #fff5ed;
    color: #75401e;
    font-size: 0.84rem;
    line-height: 1.48;
}

.limitation-title {
    margin-bottom: 6px;
    color: #a94300;
    font-weight: 800;
}


/* ------------------------------------------------------------
   STREAMLIT ELEMENTS
------------------------------------------------------------ */

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.92);
}


/* ------------------------------------------------------------
   MOBILE DISPLAY
------------------------------------------------------------ */

@media (max-width: 800px) {

    .hero-title {
        font-size: 1.8rem;
    }

    .hero-banner {
        padding: 21px;
    }

    .section-description {
        margin-left: 0;
    }

    .prediction-value {
        font-size: 1.55rem;
    }

    .prediction-icon {
        width: 48px;
        height: 48px;
    }
}

</style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD THE SAVED MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).parent
    / "optimised_random_forest_pipeline.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    deployment_bundle = load_model()
    model = deployment_bundle["model"]

except FileNotFoundError:
    st.error(
        "The saved model file could not be found. Ensure "
        "'optimised_random_forest_pipeline.joblib' is in the "
        "same folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.stop()


# ============================================================
# SIDEBAR: MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
<div class="sidebar-brand">
<div class="sidebar-brand-icon">⌂</div>
<div class="sidebar-brand-name">Sydney Property<br>Price Estimator</div>
</div>

<div class="sidebar-title">ⓘ &nbsp;Model information</div>

<div class="sidebar-label">MODEL</div>
<div class="sidebar-value">Optimised Random Forest</div>

<div class="sidebar-label">TRAINING OBSERVATIONS</div>
<div class="sidebar-value">120 Sydney property sales recorded between 20 September 2025 and 30 June 2026</div>

<div class="sidebar-label">TYPICAL MODEL ERROR</div>
<div class="sidebar-value">
Held-out test MAE: approximately AUD 404,830.<br>
Five-fold cross-validation MAE: approximately AUD 363,758.<br>
These values represent average absolute errors across evaluation properties, not an error range for an individual property.
</div>

<div class="sidebar-label">PRICE CURRENCY</div>
<div class="sidebar-value">Australian dollars (AUD)</div>

<div class="sidebar-label">NUMBER OF PREDICTORS</div>
<div class="sidebar-value">9</div>

<div class="sidebar-label">SUPPORTED SUBURBS</div>
<div class="sidebar-value">Mosman, Parramatta and Liverpool</div>

<div class="limitation-card">
<div class="limitation-title">ⓘ &nbsp;Important limitation</div>
The training dataset is small and covers only three suburbs.
Unusual properties or properties outside this scope may have
larger prediction errors.
</div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO BANNER
# ============================================================

st.markdown(
    """
<div class="hero-banner">
<div class="hero-content">
<h1 class="hero-title">Sydney Property Price Estimator</h1>
<p class="hero-subtitle">Enter the property and location details below to receive an indicative sale-price estimate from the trained machine-learning model.</p>
<span class="hero-badge">⚙️ &nbsp;Optimised Random Forest</span>
<span class="hero-badge">📊 &nbsp;120 Sydney sales</span>
</div>
</div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ABOUT THE PROJECT
# ============================================================

st.markdown(
    """
<div class="about-card">
<div class="about-icon">ⓘ</div>
<div>
<div class="about-title">About this project</div>
<p class="about-text">This application was created as part of a student machine-learning project to explore how property characteristics and location influence residential sale prices across Mosman, Parramatta and Liverpool.</p>
</div>
</div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROPERTY INPUT FORM
# ============================================================

with st.form("property_form"):

    location_column, property_column = st.columns(
        [0.82, 1.18],
        gap="medium"
    )

    # --------------------------------------------------------
    # LOCATION INPUTS
    # --------------------------------------------------------

    with location_column:

        with st.container(border=True):

            st.markdown(
                """
<div class="section-heading">
<span class="section-icon">⌖</span>
<span>Location</span>
</div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
<div class="section-description">
Enter the location characteristics of the property.
</div>
                """,
                unsafe_allow_html=True
            )

            suburb = st.selectbox(
                "Suburb",
                [
                    "Mosman",
                    "Parramatta",
                    "Liverpool"
                ]
            )

            distance_to_cbd_km = st.number_input(
                "Distance to Sydney CBD (km)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.1,
                format="%.2f"
            )

            distance_to_station_km = st.number_input(
                "Distance to nearest station (km)",
                min_value=0.0,
                max_value=10.0,
                value=2.1,
                step=0.1,
                format="%.2f"
            )

    # --------------------------------------------------------
    # PROPERTY INPUTS
    # --------------------------------------------------------

    with property_column:

        with st.container(border=True):

            st.markdown(
                """
<div class="section-heading">
<span class="section-icon">⌂</span>
<span>Property details</span>
</div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
<div class="section-description">
Enter the main characteristics of the property.
</div>
                """,
                unsafe_allow_html=True
            )

            first_column, second_column = st.columns(2)

            with first_column:

                property_type = st.selectbox(
                    "Property type",
                    [
                        "House",
                        "Apartment-Unit",
                        "Townhouse"
                    ]
                )

                bathrooms = st.number_input(
                    "Bathrooms",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1
                )

            with second_column:

                bedrooms = st.number_input(
                    "Bedrooms",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1
                )

                parking_spaces_input = st.number_input(
                    "Parking spaces",
                    min_value=0,
                    max_value=10,
                    value=1,
                    step=1
                )

            parking_unknown = st.checkbox(
                "Parking information unavailable"
            )

            land_column, land_checkbox_column = st.columns(
                [1, 1]
            )

            with land_column:

                land_size_input = st.number_input(
                    "Land size (m²)",
                    min_value=1.0,
                    max_value=3000.0,
                    value=370.0,
                    step=10.0,
                    format="%.2f"
                )

            with land_checkbox_column:

                st.markdown(
                    "<br>",
                    unsafe_allow_html=True
                )

                land_unknown = st.checkbox(
                    "Land-size information unavailable"
                )

            sale_method = st.selectbox(
                "Sale method",
                [
                    "Auction",
                    "Private Treaty",
                    "Sold prior to auction"
                ]
            )

    predict_button = st.form_submit_button(
        "Estimate sale price",
        use_container_width=True
    )


# ============================================================
# GENERATE AND DISPLAY PREDICTION
# ============================================================

if predict_button:

    parking_spaces = (
        np.nan
        if parking_unknown
        else parking_spaces_input
    )

    land_size_m2 = (
        np.nan
        if land_unknown
        else land_size_input
    )

    property_input = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "parking_spaces": [parking_spaces],
        "land_size_m2": [land_size_m2],
        "distance_to_cbd_km": [distance_to_cbd_km],
        "distance_to_station_km": [distance_to_station_km],
        "suburb": [suburb],
        "property_type": [property_type],
        "sale_method": [sale_method]
    })

    # Match the exact predictor order used during training
    property_input = property_input[
        deployment_bundle["predictor_columns"]
    ]

    try:
        estimated_price = model.predict(
            property_input
        )[0]

        st.markdown(
            f"""
        <div class="prediction-card">
        <div class="prediction-icon">
        <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M7 36L17 25L24 31L34 16L41 22" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="7" cy="36" r="2.5" fill="#ff6500" stroke="none"/>
        <circle cx="17" cy="25" r="2.5" fill="#ff6500" stroke="none"/>
        <circle cx="24" cy="31" r="2.5" fill="#ff6500" stroke="none"/>
        <circle cx="34" cy="16" r="2.5" fill="#ff6500" stroke="none"/>
        <circle cx="41" cy="22" r="2.5" fill="#ff6500" stroke="none"/>
        </svg>
        </div>
        <div>
        <div class="prediction-label">Estimated sale price</div>
        <div class="prediction-value">AUD {estimated_price:,.0f}</div>
        <div class="prediction-caption">Indicative estimate only — not a current professional property valuation.</div>
        </div>
        </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            "▦  View submitted property information",
            expanded=True
        ):

            display_input = property_input.copy()

            display_input["parking_spaces"] = (
                display_input["parking_spaces"]
                .apply(
                    lambda value:
                    "Unavailable"
                    if pd.isna(value)
                    else int(value)
                )
            )

            display_input["land_size_m2"] = (
                display_input["land_size_m2"]
                .apply(
                    lambda value:
                    "Unavailable"
                    if pd.isna(value)
                    else f"{value:,.0f}"
                )
            )

            display_input.columns = [
                column.replace("_", " ").title()
                for column in display_input.columns
            ]

            st.dataframe(
                display_input,
                hide_index=True,
                use_container_width=True
            )

    except Exception as error:
        st.error(
            f"The estimate could not be generated: {error}"
        )


# ============================================================
# FOOTNOTE
# ============================================================

st.caption(
    "Machine-learning project • Predictions are limited "
    "by the size, location coverage and available features of "
    "the training dataset."
)
