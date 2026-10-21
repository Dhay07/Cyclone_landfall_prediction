import streamlit as st
import pandas as pd
import json
import os

from predict_tcn import predict_landfall as predict_tcn
from predict_lstm import predict_landfall as predict_lstm


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Cyclone Landfall Prediction",
    page_icon="🌀",
    layout="wide"
)


# ============================================================
# Load configurations
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(BASE_DIR, "models", "tcn", "config.json"),
    "r"
) as f:
    TCN_CONFIG = json.load(f)

with open(
    os.path.join(BASE_DIR, "models", "lstm", "config.json"),
    "r"
) as f:
    LSTM_CONFIG = json.load(f)


# ============================================================
# Title
# ============================================================

st.title("🌀 Cyclone Landfall Prediction")

st.write(
    "Research-based tropical cyclone landfall prediction "
    "using machine learning models."
)


# ============================================================
# Model selection
# ============================================================

model_choice = st.selectbox(
    "Select Prediction Model",
    [
        "TCN – Landfall Prediction",
        "LSTM – Track Prediction"
    ]
)


# ============================================================
# TCN MODEL
# ============================================================

if model_choice == "TCN – Landfall Prediction":

    st.subheader("TCN – Landfall Prediction")

    st.caption(
        f"Model version: {TCN_CONFIG['model_version']}"
    )

    st.info(
        "Enter cyclone observations sequentially. "
        "The TCN model uses the required historical track "
        "and meteorological variables."
    )

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    if "tcn_track_history" not in st.session_state:
        st.session_state.tcn_track_history = []

    # --------------------------------------------------------
    # Input form
    # --------------------------------------------------------

    with st.form("tcn_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            lat = st.number_input(
                "Latitude",
                value=15.0,
                format="%.4f"
            )

            lon = st.number_input(
                "Longitude",
                value=85.0,
                format="%.4f"
            )

            direction = st.number_input(
                "Direction from Previous",
                value=0.0,
                format="%.4f"
            )

            ws = st.number_input(
                "Wind Speed",
                value=40.0,
                format="%.4f"
            )

            cp = st.number_input(
                "Central Pressure",
                value=1000.0,
                format="%.4f"
            )

        with col2:
            month = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=1
            )

            day = st.number_input(
                "Day",
                min_value=1,
                max_value=31,
                value=1
            )

            hrs_corrected = st.number_input(
                "Hours",
                value=600.0,
                format="%.1f"
            )

            duration = st.number_input(
                "Duration",
                value=0.0,
                format="%.4f"
            )

            season = st.number_input(
                "Season",
                min_value=1,
                max_value=4,
                value=4
            )

        with col3:
            nearest_lat = st.number_input(
                "Nearest Coast Latitude",
                value=15.0,
                format="%.4f"
            )

            nearest_lon = st.number_input(
                "Nearest Coast Longitude",
                value=85.0,
                format="%.4f"
            )

            distance_coast = st.number_input(
                "Distance to Nearest Coast (km)",
                value=100.0,
                format="%.4f"
            )

            days_monsoon = st.number_input(
                "Days After Monsoon",
                value=0.0,
                format="%.4f"
            )

        submitted = st.form_submit_button(
            "Add Observation"
        )

    # --------------------------------------------------------
    # Add observation
    # --------------------------------------------------------

    if submitted:

        observation = {
            "Lat": lat,
            "Lon": lon,
            "direction_from_previous": direction,
            "WS": ws,
            "CP": cp,
            "Month": month,
            "Day": day,
            "hrs_corrected": hrs_corrected,
            "Duration": duration,
            "Season": season,
            "Nearest_lat": nearest_lat,
            "Nearest_lon": nearest_lon,
            "Distance_to_nearest_coast_km": distance_coast,
            "daysaftermonsoon": days_monsoon
        }

        st.session_state.tcn_track_history.append(
            observation
        )

        st.success(
            f"Observation "
            f"{len(st.session_state.tcn_track_history)} added."
        )

    # --------------------------------------------------------
    # Display observations
    # --------------------------------------------------------

    if st.session_state.tcn_track_history:

        track_df = pd.DataFrame(
            st.session_state.tcn_track_history
        )

        st.subheader("Current Track")

        st.dataframe(
            track_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if len(track_df) >= 2:

            try:

                result = predict_tcn(track_df)

                st.subheader("Predicted Landfall")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Latitude",
                        f"{result['predicted_landfall_lat']:.4f}°"
                    )

                with col2:
                    st.metric(
                        "Longitude",
                        f"{result['predicted_landfall_lon']:.4f}°"
                    )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )

        else:

            st.info(
                "Add at least two observations to generate "
                "a prediction."
            )

        if st.button("Clear TCN Track"):

            st.session_state.tcn_track_history = []

            st.rerun()


# ============================================================
# LSTM MODEL
# ============================================================

else:

    st.subheader("LSTM – Track Prediction")

    st.caption(
        f"Model version: {LSTM_CONFIG['model_version']}"
    )

    st.info(
        "The LSTM model uses the latest two consecutive "
        "cyclone observations to predict landfall location "
        "12 hours ahead."
    )

    # --------------------------------------------------------
    # Session state
    # --------------------------------------------------------

    if "lstm_track_history" not in st.session_state:
        st.session_state.lstm_track_history = []

    # --------------------------------------------------------
    # Input form
    # --------------------------------------------------------

    with st.form("lstm_form"):

        col1, col2 = st.columns(2)

        with col1:

            lat = st.number_input(
                "Latitude",
                value=15.0,
                format="%.4f"
            )

            lon = st.number_input(
                "Longitude",
                value=85.0,
                format="%.4f"
            )

            cp = st.number_input(
                "Central Pressure",
                value=1000.0,
                format="%.4f"
            )

            ws = st.number_input(
                "Wind Speed",
                value=40.0,
                format="%.4f"
            )

            year = st.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                value=2026
            )

        with col2:

            month = st.number_input(
                "Month",
                min_value=1,
                max_value=12,
                value=1
            )

            day = st.number_input(
                "Day",
                min_value=1,
                max_value=31,
                value=1
            )

            hrs_corrected = st.number_input(
                "Hours",
                value=600.0,
                format="%.1f"
            )

            gradeno = st.number_input(
                "Grade Number",
                value=1.0,
                format="%.4f"
            )

            season = st.number_input(
                "Season",
                min_value=1,
                max_value=4,
                value=4
            )

        submitted = st.form_submit_button(
            "Add Observation"
        )

    # --------------------------------------------------------
    # Add observation
    # --------------------------------------------------------

    if submitted:

        observation = {
            "Lat": lat,
            "Lon": lon,
            "CP": cp,
            "WS": ws,
            "Year": year,
            "Month": month,
            "Day": day,
            "hrs_corrected": hrs_corrected,
            "Gradeno": gradeno,
            "season": season
        }

        st.session_state.lstm_track_history.append(
            observation
        )

        st.success(
            f"Observation "
            f"{len(st.session_state.lstm_track_history)} added."
        )

    # --------------------------------------------------------
    # Display observations
    # --------------------------------------------------------

    if st.session_state.lstm_track_history:

        track_df = pd.DataFrame(
            st.session_state.lstm_track_history
        )

        st.subheader("Current Track")

        st.dataframe(
            track_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # LSTM prediction
        # ----------------------------------------------------

        if len(track_df) >= LSTM_CONFIG["window_size"]:

            try:

                result = predict_lstm(track_df)

                st.subheader(
                    f"Predicted track "
                    f"({LSTM_CONFIG['prediction_horizon_hours']}-hour horizon)"
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Latitude",
                        f"{result['predicted_lat']:.4f}°"
                    )

                with col2:
                    st.metric(
                        "Longitude",
                        f"{result['predicted_lon']:.4f}°"
                    )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )

        else:

            st.info(
                "Add two observations to generate the "
                "first LSTM prediction."
            )

        if st.button("Clear LSTM Track"):

            st.session_state.lstm_track_history = []

            st.rerun()


# ============================================================
# Disclaimer
# ============================================================

st.divider()

st.caption(
    "This application presents predictions from research-based "
    "machine learning models. It is not an official cyclone "
    "warning or forecasting service."
)