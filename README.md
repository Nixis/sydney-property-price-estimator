# Sydney Property Price Estimator

A machine-learning project developed for the Deakin University **SIT720 Machine Learning – 8.1 Distinction Task**.

The project predicts residential sale prices for properties in **Mosman, Parramatta and Liverpool, NSW**. It covers the complete machine-learning workflow, including manual data collection, exploratory analysis, feature engineering, model comparison, prediction-failure analysis and deployment through Streamlit.

## Live Application

Use the deployed application here:

[Open the Sydney Property Price Estimator](https://nixis-sydney-property-estimator.streamlit.app/)

The application generates indicative sale-price estimates only. It is not a substitute for a professional property valuation.

## Dataset

The dataset contains **120 residential property sales**:

- 40 properties from Mosman
- 40 properties from Parramatta
- 40 properties from Liverpool

Each suburb contains:

- 16 houses
- 16 apartments/units
- 8 townhouses

Properties were manually collected from publicly available Domain.com.au sold-property listings. The sales occurred between **20 September 2025 and 30 June 2026**.

The dataset contains property and sale information such as:

- Suburb
- Sale price
- Sale date
- Property type
- Bedrooms
- Bathrooms
- Parking spaces
- Land size
- Sale method
- Property description
- Source listing URL

## Feature Engineering

Two location-related predictors were engineered:

- Straight-line distance to the Sydney CBD
- Straight-line distance to the nearest train or metro station

Property-level coordinates were researched and verified before calculating these distances using the Haversine formula. Public transport station information was obtained from the Transport for NSW GTFS dataset.

## Machine-Learning Models

Three regression approaches were developed and compared:

1. Multiple Linear Regression
2. Random Forest Regression
3. AdaBoost Regression

A Median Baseline was also included as a minimum-performance benchmark.

Models were evaluated using five-fold cross-validation on the training data. The **Optimised Random Forest** achieved the strongest cross-validation performance and was selected for deployment.

### Selected Model Performance

| Evaluation | MAE | RMSE | R² |
|---|---:|---:|---:|
| Five-fold cross-validation | AUD 363,758 | AUD 568,258 | 0.868 |
| Held-out test set | AUD 404,830 | AUD 591,840 | 0.656 |

The held-out test set contained 24 properties that were not used during model training, preprocessing or hyperparameter selection.

## Model Inputs

The deployed model uses nine predictors:

1. Suburb
2. Property type
3. Bedrooms
4. Bathrooms
5. Parking spaces
6. Land size
7. Sale method
8. Distance to the Sydney CBD
9. Distance to the nearest station

Missing parking and land-size values are handled using property-type-specific median imputation learned during model fitting.

## Repository Contents

| File | Description |
|---|---|
| `app.py` | Streamlit web-application source code |
| `optimised_random_forest_pipeline.joblib` | Saved preprocessing and Random Forest pipeline |
| `requirements.txt` | Required Python package versions |
| `sydney_property_sales_engineered_final.csv` | Final model-ready dataset with engineered features |
| `property_sales_verified_coordinates_final.csv` | Dataset containing verified property coordinates |
| `sydney_property_sales_original.csv` | Original manually collected property dataset |
| `8.1D-Code.ipynb` | Jupyter notebook containing the complete analysis and model development |
| `README.md` | Project overview and usage instructions |

> Some filenames may differ depending on the final uploaded files.

## Run the Application Locally

### 1. Download the project

Clone the repository:

```bash
git clone https://github.com/Nixis/sydney-property-price-estimator.git
