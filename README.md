# Sydney Property Price Estimator

A machine-learning application that provides indicative residential sale-price estimates for properties in **Mosman, Parramatta and Liverpool, NSW**.

## Live Application

[Open the Sydney Property Price Estimator](https://nixis-sydney-property-estimator.streamlit.app/)

## Project Overview

The project uses an Optimised Random Forest model trained on 120 sold properties:

- 40 properties from each suburb
- Houses, apartments/units and townhouses
- Sales recorded between 20 September 2025 and 30 June 2026

The model uses property characteristics, location, sale method, distance to the Sydney CBD and distance to the nearest train or metro station.

## Repository Files

| File | Description |
|---|---|
| `app.py` | Streamlit application |
| `optimised_random_forest_pipeline.joblib` | Saved model and preprocessing pipeline |
| `requirements.txt` | Required Python packages |
| `sydney_property_sales_engineered_final.csv` | Final dataset with engineered features |

## Run Locally

```bash
git clone https://github.com/Nixis/sydney-property-price-estimator.git
cd sydney-property-price-estimator
pip install -r requirements.txt
streamlit run app.py
