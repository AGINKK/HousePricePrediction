
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"

st.title("🏡 House Price Prediction")
st.write(
    "Predict the median house value using a Machine Learning model "
    "trained on the California Housing dataset."
)

st.divider()

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please save your trained model at "
        "models/house_price_model.pkl before running this app."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter Housing Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Median Income (MedInc)",
            min_value=0.0,
            value=3.5,
            step=0.1,
            help="Median household income in units of $10,000."
        )

        house_age = st.number_input(
            "House Age (HouseAge)",
            min_value=0.0,
            value=25.0,
            step=1.0
        )

        ave_rooms = st.number_input(
            "Average Rooms (AveRooms)",
            min_value=0.0,
            value=5.0,
            step=0.1
        )

        ave_bedrms = st.number_input(
            "Average Bedrooms (AveBedrms)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

    with col2:
        population = st.number_input(
            "Population",
            min_value=0.0,
            value=1000.0,
            step=10.0
        )

        ave_occup = st.number_input(
            "Average Occupancy (AveOccup)",
            min_value=0.0,
            value=3.0,
            step=0.1
        )

        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=34.05,
            step=0.01
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.24,
            step=0.01
        )

    submitted = st.form_submit_button(
        "Predict House Value",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame(
        [[
            med_inc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude
        ]],
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]
    )

    try:
        prediction = float(model.predict(input_data)[0])

        st.divider()
        st.subheader("Prediction Result")

        if prediction < 0:
            st.warning(
                "The model predicted a negative value. "
                "This may indicate an unreliable prediction for these inputs."
            )
        else:
            # Target values are measured in units of $100,000.
            predicted_dollars = prediction * 100000

            st.success("Prediction completed successfully!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Median House Value",
                    f"${predicted_dollars:,.0f}"
                )

            with col2:
                st.metric(
                    "Model Output",
                    f"{prediction:.3f}"
                )

            st.caption(
                "The dataset target is measured in units of $100,000. "
                "This is an estimated median value for a California district, "
                "not an exact price for an individual house."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

st.caption(
    "Educational Machine Learning Project | "
    "Model: Linear Regression | Dataset: California Housing"
)
