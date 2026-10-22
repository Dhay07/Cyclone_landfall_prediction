import streamlit as st
import pandas as pd
import json
import os
import calendar
from datetime import datetime
from zoneinfo import ZoneInfo

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
# 3. DATE AND SEASON SETTINGS
# ============================================================

# Current date in Indian Standard Time
today = datetime.now(ZoneInfo("Asia/Kolkata"))

CURRENT_YEAR = today.year
CURRENT_MONTH = today.month
CURRENT_DAY = today.day

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

SEASON_NAMES = {
    1: "Winter (Jan–Feb)",
    2: "Summer (Mar–May)",
    3: "Southwest Monsoon (Jun–Sep)",
    4: "Northeast Monsoon (Oct–Dec)"
}


def get_season(month):
    """Return the seasonal category based on the selected month."""
    if month in (1, 2):
        return 1
    elif month in (3, 4, 5):
        return 2
    elif month in (6, 7, 8, 9):
        return 3
    else:
        return 4


# ============================================================
# 4. INPUT INFORMATION POPOVER
# ============================================================

def add_input(label, key, explanation, **kwargs):
    """Display a numeric input with an adjacent information popover."""

    input_col, info_col = st.columns([3, 0.5])

    with input_col:
        value = st.number_input(
            label,
            key=key,
            **kwargs
        )

    with info_col:
        with st.popover("ⓘ"):
            st.write(explanation)

    return value


# ============================================================
# 5. ABOUT DIALOG
# ============================================================

@st.dialog("About the Prediction Models")
def show_about():

    st.markdown("""
    This application features two deep-learning models for
    tropical cyclone prediction: Long Short-Term Memory (LSTM)
    and Temporal Convolutional Network (TCN).

    ### Data and Methodology

    Both models were developed using historical tropical cyclone
    Best Track data published by the India Meteorological
    Department (IMD) for 1990–2023. Additional variables were
    derived from historical cyclone records. Seasonal information
    is included to represent seasonal variations in cyclone
    characteristics.

    The models focus on tropical cyclones over the Bay of Bengal.

    ### Model Predictions

    - **TCN — Landfall Prediction:** Estimates the cyclone's
      landfall location using the supplied observations and
      TCN-specific input variables.

    - **LSTM — Track Prediction:** Predicts the cyclone's
      position 12 hours ahead using the latest two consecutive
      observations.

    Both models receive observations from the same input page,
    but their inputs are prepared and processed separately.

    ### Data Requirements and Limitations

    - These models do not predict cyclone formation.
    - At least two observations at six-hour intervals are required
      for the predictions.
    - Predictions depend on the accuracy and consistency of
      the supplied observations.
    - The models are primarily intended for the Bay of Bengal;
      performance may vary across cyclones and seasons.

    Current cyclone observations may be obtained from
    [Windy](https://www.windy.com/) and the
    [India Meteorological Department](https://mausam.imd.gov.in/).

    ---

    **Disclaimer:** This application is intended for research
    and educational purposes only. Its predictions are not
    official forecasts or warnings and must not be used as a
    substitute for forecasts, advisories, or warnings issued by
    the IMD or other authorized meteorological agencies.
    """)


# ============================================================
# 6. TITLE
# ============================================================

st.title("🌀 Cyclone Landfall Prediction")

st.write(
    "Research-based tropical cyclone prediction using "
    "TCN and LSTM deep-learning models."
)

if st.button("ℹ️ About the Models"):
    show_about()

st.divider()


# ============================================================
# 7. INITIALIZE SEPARATE MODEL HISTORIES
# ============================================================

if "tcn_track_history" not in st.session_state:
    st.session_state.tcn_track_history = []

if "lstm_track_history" not in st.session_state:
    st.session_state.lstm_track_history = []


# ============================================================
# 8. COMMON INPUTS
# ============================================================

st.header("Cyclone Observation")

st.caption(
    "Enter one cyclone observation at a time. The observation "
    "will be added to both model histories, with model-specific "
    "variables processed separately."
)

st.subheader("Date and Time")

date_col1, date_col2, date_col3, date_col4 = st.columns(4)

with date_col1:
    year = st.selectbox(
        "Year",
        options=list(range(1990, CURRENT_YEAR + 1)),
        index=CURRENT_YEAR - 1990,
        key="obs_year"
    )

with date_col2:
    month_display = st.selectbox(
        "Month",
        options=MONTH_NAMES,
        index=CURRENT_MONTH - 1,
        key="obs_month"
    )

month = MONTH_NAMES.index(month_display) + 1

with date_col3:
    max_day = calendar.monthrange(year, month)[1]

    day = st.selectbox(
        "Day",
        options=list(range(1, max_day + 1)),
        index=min(CURRENT_DAY, max_day) - 1,
        key="obs_day"
    )

with date_col4:
    hours_display = st.selectbox(
        "Time (24-hour format)",
        options=[6, 12, 18, 24],
        index=0,
        format_func=lambda x: f"{x:02d}:00",
        key="obs_hours"
    )

# Both existing models use the HHMM-style numeric representation.
hrs_corrected = float(hours_display * 100)

# Derive season automatically from the selected month.
season = get_season(month)

# Derive season automatically from the selected month.
season = get_season(month)

SEASON_INFO = {
    1: {
        "name": "Winter",
        "description": (
            "A rare period for tropical cyclones to occur due to "
            "generally unfavourable atmospheric conditions."
        )
    },
    2: {
        "name": "Summer (Pre-monsoon)",
        "description": (
            "The pre-monsoon period has produced some devastating "
            "cyclones in the past, significantly affecting India."
        )
    },
    3: {
        "name": "Southwest Monsoon (SWM)",
        "description": (
            "The monsoon period produces relatively more depressions "
            "and deep depressions, many of which do not intensify "
            "into cyclonic storms."
        )
    },
    4: {
        "name": "Northeast Monsoon (NEM)",
        "description": (
            "The post-monsoon period is associated with many of the "
            "high-intensity cyclones affecting India, particularly "
            "along the east coast."
        )
    }
}

season_name = SEASON_INFO[season]["name"]
season_description = SEASON_INFO[season]["description"]

# Billboard-style season display
st.markdown(
    f"""
    <div style="
        border: 2px solid #4682B4;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0 20px 0;
        background-color: rgba(70, 130, 180, 0.10);
        text-align: center;
    ">
        <div style="
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 8px;
        ">
            Automatically Identified Cyclone Season
        </div>
        <div style="
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
        ">
            {season_name}
        </div>
        <div style="
            font-size: 16px;
            line-height: 1.6;
        ">
            {season_description}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("Cyclone Parameters")

common_col1, common_col2, common_col3 = st.columns(3)

with common_col1:
    lat = add_input(
        "Latitude (°)",
        "obs_lat",
        "Latitude of the cyclone centre in decimal degrees.",
        min_value=-90.0,
        max_value=90.0,
        value=15.0,
        format="%.4f"
    )

    lon = add_input(
        "Longitude (°)",
        "obs_lon",
        "Longitude of the cyclone centre in decimal degrees.",
        min_value=-180.0,
        max_value=180.0,
        value=85.0,
        format="%.4f"
    )

with common_col2:
    ws = add_input(
        "Wind Speed (knots)",
        "obs_ws",
        "Maximum sustained wind speed at the observation time, "
        "in knots.",
        min_value=0.0,
        value=40.0,
        format="%.4f"
    )

    cp = add_input(
        "Central Pressure (hPa)",
        "obs_cp",
        "Estimated minimum central pressure at the observation "
        "time, in hectopascals.",
        min_value=0.0,
        value=1000.0,
        format="%.4f"
    )

with common_col3:
    st.metric(
        "Selected observation date",
        f"{day:02d} {month_display} {year}"
    )

    st.metric(
        "Observation time",
        f"{hours_display:02d}:00"
    )

st.caption(
    "Refer IMD notification to select the time (UTC) "
    "and current cyclone postion (decimal degreees)."
)

# ============================================================
# 9. TCN-SPECIFIC INPUTS
# ============================================================

st.divider()
st.header("Additional Inputs for TCN")

st.caption(
    "These variables are used by the TCN model only. "
    "They do not replace or modify the shared inputs."
)

tcn_col1, tcn_col2, tcn_col3 = st.columns(3)

with tcn_col1:
    direction = add_input(
        "Direction from Previous (degrees)",
        "tcn_direction",
        "Direction of cyclone movement relative to the previous "
        "observation, in degrees. Use the same convention as "
        "the TCN training data.",
        min_value=0.0,
        max_value=360.0,
        value=0.0,
        format="%.4f"
    )

    duration = add_input(
        "Duration (hours)",
        "tcn_duration",
        "Elapsed cyclone duration in hours, using the same "
        "definition as the TCN training data.",
        min_value=0.0,
        value=0.0,
        format="%.4f"
    )

with tcn_col2:
    nearest_lat = add_input(
        "Nearest Coast Latitude (°)",
        "tcn_nearest_lat",
        "Latitude of the nearest coastline reference point, "
        "in decimal degrees.",
        min_value=-90.0,
        max_value=90.0,
        value=15.0,
        format="%.4f"
    )

    nearest_lon = add_input(
        "Nearest Coast Longitude (°)",
        "tcn_nearest_lon",
        "Longitude of the nearest coastline reference point, "
        "in decimal degrees.",
        min_value=-180.0,
        max_value=180.0,
        value=85.0,
        format="%.4f"
    )

with tcn_col3:
    distance_coast = add_input(
        "Distance to Nearest Coast (km)",
        "tcn_distance_coast",
        "Distance from the cyclone centre to the nearest coastline, "
        "in kilometres.",
        min_value=0.0,
        value=100.0,
        format="%.4f"
    )

    days_monsoon = add_input(
        "Days After Monsoon Onset",
        "tcn_days_monsoon",
        "Number of days between Kerala southwest monsoon onset "
        "and cyclone genesis. Use the same sign convention as "
        "the TCN training data.",
        value=0.0,
        format="%.4f"
    )


# ============================================================
# 10. LSTM-SPECIFIC INPUTS
# ============================================================

st.divider()
st.header("Additional Inputs for LSTM")

st.caption(
    "These variables are used by the LSTM model only."
)

lstm_col1, lstm_col2 = st.columns(2)

with lstm_col1:
    gradeno = add_input(
        "Grade Number",
        "lstm_gradeno",
        "Cyclone grade or intensity category as defined in "
        "the LSTM training data.",
        value=1.0,
        format="%.4f"
    )

with lstm_col2:
    st.info(
        "The LSTM uses the shared date, time, latitude, longitude, "
        "wind speed, central pressure and automatically determined "
        "season, together with the grade number."
    )


# ============================================================
# 11. ADD OBSERVATION
# ============================================================

st.divider()

submitted = st.button(
    "Add Observation and Update Predictions",
    type="primary",
    use_container_width=True
)

if submitted:

    # Validate the selected date.
    # The day dropdown is constrained by month/year, but this
    # check protects against inconsistent widget state.
    days_in_month = calendar.monthrange(year, month)[1]

    if not 1 <= day <= days_in_month:
        st.error(
            f"Invalid date: {month_display} {year} has "
            f"{days_in_month} days. Please select a valid date."
        )

    else:

        # ----------------------------------------------------
        # Prepare TCN input independently
        # ----------------------------------------------------

        tcn_observation = {
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

        # ----------------------------------------------------
        # Prepare LSTM input independently
        # ----------------------------------------------------

        lstm_observation = {
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

        # Save the same observation separately for each model.
        st.session_state.tcn_track_history.append(tcn_observation)
        st.session_state.lstm_track_history.append(lstm_observation)

        st.success(
            f"Observation added for {day:02d} {month_display} "
            f"{year}, {hours_display:02d}:00."
        )


# ============================================================
# 12. DISPLAY SHARED OBSERVATION HISTORY
# ============================================================

st.divider()
st.header("Observation History")

history_col1, history_col2 = st.columns(2)

with history_col1:
    st.subheader("TCN Input History")

    if st.session_state.tcn_track_history:
        tcn_track_df = pd.DataFrame(
            st.session_state.tcn_track_history
        )

        st.dataframe(
            tcn_track_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No observations have been added to TCN yet.")

with history_col2:
    st.subheader("LSTM Input History")

    if st.session_state.lstm_track_history:
        lstm_track_df = pd.DataFrame(
            st.session_state.lstm_track_history
        )

        st.dataframe(
            lstm_track_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No observations have been added to LSTM yet.")


# ============================================================
# 13. TCN LANDfall PREDICTION
# ============================================================

st.divider()
st.header("TCN — Landfall Prediction")

st.caption(
    f"Model version: {TCN_CONFIG['model_version']}"
)

tcn_track_df = pd.DataFrame(
    st.session_state.tcn_track_history
)

if len(tcn_track_df) >= 2:

    try:
        tcn_result = predict_tcn(tcn_track_df)

        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.metric(
                "Predicted Landfall Latitude",
                f"{tcn_result['predicted_landfall_lat']:.4f}°"
            )

        with pred_col2:
            st.metric(
                "Predicted Landfall Longitude",
                f"{tcn_result['predicted_landfall_lon']:.4f}°"
            )

    except Exception as e:
        st.error(f"TCN prediction error: {e}")

else:
    st.info(
        "Add at least two cyclone observations to generate "
        "the TCN landfall prediction."
    )


# ============================================================
# 14. LSTM TRACK PREDICTION
# ============================================================

st.divider()
st.header("LSTM — Track Prediction")

st.caption(
    f"Model version: {LSTM_CONFIG['model_version']}"
)

lstm_track_df = pd.DataFrame(
    st.session_state.lstm_track_history
)

if len(lstm_track_df) >= LSTM_CONFIG["window_size"]:

    try:
        lstm_result = predict_lstm(lstm_track_df)

        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.metric(
                "Predicted Track Latitude",
                f"{lstm_result['predicted_lat']:.4f}°"
            )

        with pred_col2:
            st.metric(
                "Predicted Track Longitude",
                f"{lstm_result['predicted_lon']:.4f}°"
            )

        st.caption(
            "Prediction horizon: "
            f"{LSTM_CONFIG['prediction_horizon_hours']} hours ahead."
        )

    except Exception as e:
        st.error(f"LSTM prediction error: {e}")

else:
    st.info(
        "Add at least two cyclone observations to generate "
        "the LSTM track prediction."
    )


# ============================================================
# 15. CLEAR OBSERVATION HISTORIES
# ============================================================

st.divider()
st.subheader("Reset Observations")

clear_col1, clear_col2 = st.columns(2)

with clear_col1:
    if st.button("Clear TCN History", use_container_width=True):
        st.session_state.tcn_track_history = []
        st.rerun()

with clear_col2:
    if st.button("Clear LSTM History", use_container_width=True):
        st.session_state.lstm_track_history = []
        st.rerun()


# ============================================================
# 16. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This application presents predictions from research-based "
    "machine-learning models. It is not an official cyclone "
    "warning or forecasting service."
)