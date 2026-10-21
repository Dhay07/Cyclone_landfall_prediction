
import streamlit as st
import pandas as pd
import json
import os

from predict_tcn import predict_landfall as predict_tcn
from predict_lstm import predict_landfall as predict_lstm


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cyclone Landfall Prediction",
    page_icon="🌀",
    layout="wide"
)


# ============================================================
# 2. LOAD CONFIGURATIONS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(BASE_DIR, "models", "tcn", "config.json"),
    "r",
    encoding="utf-8"
) as f:
    TCN_CONFIG = json.load(f)

with open(
    os.path.join(BASE_DIR, "models", "lstm", "config.json"),
    "r",
    encoding="utf-8"
) as f:
    LSTM_CONFIG = json.load(f)


# ============================================================
# 3. INPUT INFORMATION POPOVER
# ============================================================

def add_input(label, key, explanation, **kwargs):
    """Display a numeric input with an adjacent information popover."""

    input_col, info_col = st.columns([5, 1])

    with input_col:
        value = st.number_input(
            label,
            key=key,
            **kwargs
        )

    with info_col:
        with st.popover("ⓘ", help=f"Information about {label}"):
            st.markdown(f"**{label}**")
            st.write(explanation)

    return value


# ============================================================
# 4. ABOUT DIALOG
# ============================================================

@st.dialog("About the Prediction Models")
def show_about():

    st.markdown(
        """
        This application features two deep learning models
        for tropical cyclone prediction: Long Short-Term
        Memory (LSTM) and Temporal Convolutional Network
        (TCN).

        The *LSTM model* predicts the cyclone's track
        12 hours ahead using sequential historical
        observations. In contrast, the *sequence-to-one
        TCN model* directly predicts the cyclone's
        landfall location from the provided track and
        meteorological information.

        As the models use different input variables and
        prediction approaches, select a model and provide
        the corresponding inputs accordingly.

        Use Windy or the official IMD website to obtain
        cyclone observations. Ensure that the units,
        timestamps, and variable definitions match the
        model's requirements.

        ---

        **Disclaimer**

        This application is intended for educational and
        research purposes only. Its predictions are not
        official forecasts or warnings and should not be
        used as a substitute for guidance from authorized
        meteorological agencies.
        """
    )


# ============================================================
# 5. TITLE AND ABOUT BUTTON
# ============================================================

st.title("🌀 Cyclone Landfall Prediction")

st.write(
    "Research-based tropical cyclone landfall prediction "
    "using machine learning models."
)

if st.button("ℹ️ About the Models"):
    show_about()


# ============================================================
# 6. MODEL SELECTION
# ============================================================

model_choice = st.selectbox(
    "Select Prediction Model",
    [
        "TCN – Landfall Prediction",
        "LSTM – Track Prediction"
    ]
)


# ============================================================
# 7. TCN MODEL
# ============================================================

if model_choice == "TCN – Landfall Prediction":

    st.subheader("TCN – Landfall Prediction")

    st.caption(
        f"Model version: {TCN_CONFIG['model_version']}"
    )

    st.info(
        "Enter or update cyclone observations sequentially "
        "at 6-hour intervals. The TCN model uses the "
        "provided observations to predict landfall."
    )

    if "tcn_track_history" not in st.session_state:
        st.session_state.tcn_track_history = []

    with st.form("tcn_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            lat = add_input(
                "Latitude", "tcn_lat",
                "Latitude of the cyclone centre in decimal degrees. "
                "Positive values indicate locations north of the Equator; "
                "negative values indicate locations south of it.",
                value=15.0, format="%.4f"
            )

            lon = add_input(
                "Longitude", "tcn_lon",
                "Longitude of the cyclone centre in decimal degrees. "
                "Use the same longitude convention as your training data.",
                value=85.0, format="%.4f"
            )

            direction = add_input(
                "Direction from Previous", "tcn_direction",
                "Direction of cyclone movement relative to the previous "
                "observation. Use the same units and directional convention "
                "as the model's training dataset.",
                value=0.0, format="%.4f"
            )

            ws = add_input(
                "Wind Speed", "tcn_ws",
                "Maximum sustained wind speed at the observation time. "
                "Enter it in the same units used during model training.",
                value=40.0, format="%.4f"
            )

            cp = add_input(
                "Central Pressure", "tcn_cp",
                "Estimated central pressure of the cyclone. Use the same "
                "pressure units as the model's training data.",
                value=1000.0, format="%.4f"
            )

        with col2:

            month = add_input(
                "Month", "tcn_month",
                "Calendar month of the observation: January = 1 through "
                "December = 12.",
                min_value=1, max_value=12, value=1
            )

            day = add_input(
                "Day", "tcn_day",
                "Calendar day of the observation, from 1 to 31. "
                "Ensure the day corresponds to the observation date.",
                min_value=1, max_value=31, value=1
            )

            hrs_corrected = add_input(
                "Hours", "tcn_hours",
                "The time variable named hrs_corrected in the training "
                "dataset. Use the same reference time and calculation "
                "method as the training data; do not enter clock time "
                "unless that is how the variable was defined.",
                value=600.0, format="%.1f"
            )

            duration = add_input(
                "Duration", "tcn_duration",
                "Cyclone duration variable as defined in the training "
                "dataset. Use the same time units and reference point.",
                value=0.0, format="%.4f"
            )

            season = add_input(
                "Season", "tcn_season",
                "Seasonal category encoded using the model's training "
                "convention. Select the category corresponding to the "
                "observation period; do not assume a different encoding.",
                min_value=1, max_value=4, value=4
            )

        with col3:

            nearest_lat = add_input(
                "Nearest Coast Latitude", "tcn_nearest_lat",
                "Latitude of the nearest coastline reference point, "
                "in decimal degrees. Follow the method used to derive "
                "this feature during model training.",
                value=15.0, format="%.4f"
            )

            nearest_lon = add_input(
                "Nearest Coast Longitude", "tcn_nearest_lon",
                "Longitude of the nearest coastline reference point, "
                "in decimal degrees. Follow the method used during "
                "model training.",
                value=85.0, format="%.4f"
            )

            distance_coast = add_input(
                "Distance to Nearest Coast (km)", "tcn_distance_coast",
                "Distance from the cyclone centre to the nearest coastline, "
                "in kilometres. Use the same distance calculation method "
                "as the training dataset.",
                value=100.0, format="%.4f"
            )

            days_monsoon = add_input(
                "Days After Monsoon", "tcn_days_monsoon",
                "Number of days relative to the monsoon-onset reference "
                "used in training. Calculate this using the same onset "
                "date and sign convention as the training dataset.",
                value=0.0, format="%.4f"
            )

        submitted = st.form_submit_button("Add Observation")

    # --------------------------------------------------------
    # ADD OBSERVATION
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

        st.session_state.tcn_track_history.append(observation)

        st.success(
            f"Observation "
            f"{len(st.session_state.tcn_track_history)} added."
        )

    # --------------------------------------------------------
    # DISPLAY OBSERVATIONS AND PREDICT
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
                st.error(f"Prediction error: {e}")

        else:
            st.info(
                "Add at least two observations to generate "
                "a prediction."
            )

        if st.button("Clear TCN Track"):
            st.session_state.tcn_track_history = []
            st.rerun()


# ============================================================
# 8. LSTM MODEL
# ============================================================

else:

    st.subheader("LSTM – Track Prediction")

    st.caption(
        f"Model version: {LSTM_CONFIG['model_version']}"
    )

    st.info(
        "Enter or update cyclone observations sequentially "
        "at 6-hour intervals. The LSTM model uses the latest "
        "two consecutive observations to predict the cyclone "
        "track 12 hours ahead."
    )

    if "lstm_track_history" not in st.session_state:
        st.session_state.lstm_track_history = []

    with st.form("lstm_form"):

        col1, col2 = st.columns(2)

        with col1:

            lat = add_input(
                "Latitude", "lstm_lat",
                "Latitude of the cyclone centre in decimal degrees. "
                "Positive values indicate locations north of the Equator.",
                value=15.0, format="%.4f"
            )

            lon = add_input(
                "Longitude", "lstm_lon",
                "Longitude of the cyclone centre in decimal degrees. "
                "Use the same longitude convention as the training data.",
                value=85.0, format="%.4f"
            )

            cp = add_input(
                "Central Pressure", "lstm_cp",
                "Estimated central pressure of the cyclone. Enter the "
                "same pressure units used in model training.",
                value=1000.0, format="%.4f"
            )

            ws = add_input(
                "Wind Speed", "lstm_ws",
                "Maximum sustained wind speed at the observation time. "
                "Use the same units as the training dataset.",
                value=40.0, format="%.4f"
            )

            year = add_input(
                "Year", "lstm_year",
                "Calendar year of the observation.",
                min_value=1900, max_value=2100, value=2026
            )

        with col2:

            month = add_input(
                "Month", "lstm_month",
                "Calendar month of the observation: January = 1 "
                "through December = 12.",
                min_value=1, max_value=12, value=1
            )

            day = add_input(
                "Day", "lstm_day",
                "Calendar day of the observation, from 1 to 31.",
                min_value=1, max_value=31, value=1
            )

            hrs_corrected = add_input(
                "Hours", "lstm_hours",
                "The hrs_corrected time variable used during training. "
                "Enter it according to the same reference time and "
                "calculation method as the training dataset.",
                value=600.0, format="%.1f"
            )

            gradeno = add_input(
                "Grade Number", "lstm_gradeno",
                "Cyclone grade or intensity category represented by "
                "the Grade Number variable. Use the category encoding "
                "adopted in the training dataset.",
                value=1.0, format="%.4f"
            )

            season = add_input(
                "Season", "lstm_season",
                "Seasonal category encoded according to the LSTM "
                "training convention. Use the correct category for "
                "the observation period.",
                min_value=1, max_value=4, value=4
            )

        submitted = st.form_submit_button("Add Observation")

    # --------------------------------------------------------
    # ADD OBSERVATION
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

        st.session_state.lstm_track_history.append(observation)

        st.success(
            f"Observation "
            f"{len(st.session_state.lstm_track_history)} added."
        )

    # --------------------------------------------------------
    # DISPLAY OBSERVATIONS AND PREDICT
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

        if len(track_df) >= LSTM_CONFIG["window_size"]:

            try:

                result = predict_lstm(track_df)

                st.subheader(
                    f"Predicted Track "
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
                st.error(f"Prediction error: {e}")

        else:
            st.info(
                "Add two observations to generate the "
                "first LSTM prediction."
            )

        if st.button("Clear LSTM Track"):
            st.session_state.lstm_track_history = []
            st.rerun()


# ============================================================
# 9. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This application presents predictions from research-based "
    "machine learning models. It is not an official cyclone "
    "warning or forecasting service."
)
