import streamlit as st
import pandas as pd
import json
import os
import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math

# Optional geospatial dependencies are required for automatic coastline features.
try:
    import geopandas as gpd
    from pyproj import CRS
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    GEOSPATIAL_IMPORT_ERROR = None
except ImportError as exc:
    gpd = None
    CRS = None
    Point = None
    nearest_points = None
    GEOSPATIAL_IMPORT_ERROR = str(exc)

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
# AUTOMATED TCN FEATURE HELPERS
# ============================================================

# Set COASTLINE_PATH to your coastline file, or place the file at
# <app folder>/data/coastline.shp. GeoJSON, Shapefile (.shp), and
# other vector formats supported by GeoPandas can be used.
COASTLINE_PATH = os.environ.get(
    "COASTLINE_PATH",
    os.path.join(BASE_DIR, "data", "coastline.shp")
)


@st.cache_resource
def load_coastline_data(path, modified_time):
    """Load coastline vector data and normalize it to longitude/latitude."""
    if gpd is None:
        return None, (
            "Automatic coastline calculations require geopandas, shapely, "
            "and pyproj. Install them with: pip install geopandas shapely pyproj."
        )
    if not os.path.exists(path):
        return None, (
            f"Coastline file not found: {path}. Set COASTLINE_PATH or place "
            "your coastline file at data/coastline.geojson."
        )
    try:
        coast = gpd.read_file(path)
        if coast.empty:
            return None, "The coastline dataset contains no features."
        if coast.crs is None:
            return None, (
                "The coastline dataset has no declared CRS. Define its actual "
                "coordinate reference system before using it."
            )
        coast = coast.to_crs("EPSG:4326")
        coast = coast.loc[coast.geometry.notna() & ~coast.geometry.is_empty]
        if coast.empty:
            return None, "The coastline dataset has no usable geometries."
        return coast, None
    except Exception as exc:
        return None, f"Could not read coastline dataset: {exc}"


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from the previous point to the current point.

    Returns degrees clockwise from true north in the range [0, 360).
    Confirm that this is the same direction convention used in TCN training.
    """
    if math.isclose(lat1, lat2, abs_tol=1e-12) and math.isclose(
        lon1, lon2, abs_tol=1e-12
    ):
        return 0.0

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    x = math.sin(delta_lon) * math.cos(phi2)
    y = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def find_nearest_coast(lat, lon, coastline):
    """Return (nearest coastline latitude, longitude, distance in km).

    A local azimuthal-equidistant projection is used so distance is measured
    in metres rather than degrees. A local geographic window speeds up the
    calculation for global coastline datasets.
    """
    if coastline is None or coastline.empty:
        raise ValueError("Coastline data is not loaded.")

    # Use nearby coastline features when available; otherwise use the full set.
    local = coastline.cx[lon - 15:lon + 15, lat - 15:lat + 15]
    if local.empty:
        local = coastline

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} "
        "+datum=WGS84 +units=m +no_defs"
    )
    cyclone_point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(local_crs).iloc[0]
    coast_projected = local.to_crs(local_crs)
    coast_union = coast_projected.geometry.unary_union
    _, nearest_point = nearest_points(cyclone_point, coast_union)
    distance_km = cyclone_point.distance(nearest_point) / 1000.0
    nearest_lonlat = gpd.GeoSeries([nearest_point], crs=local_crs).to_crs("EPSG:4326").iloc[0]
    return float(nearest_lonlat.y), float(nearest_lonlat.x), float(distance_km)


def make_observation_datetime(year, month, day, hour_utc):
    """Build a UTC timestamp; the UI's 24:00 option rolls to next-day 00:00."""
    return datetime(year, month, day) + timedelta(hours=int(hour_utc))


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
    - Automated TCN coastline features require a georeferenced coastline
      vector dataset; direction calculations must match the training convention.
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

st.title("🌀 Cyclone Landfall Prediction - Bay of Bengal")

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

# Timestamps are stored separately so no helper columns are added to the
# DataFrame passed into the trained TCN model.
if "tcn_first_observation_datetime" not in st.session_state:
    st.session_state.tcn_first_observation_datetime = None
if "tcn_last_observation_datetime" not in st.session_state:
    st.session_state.tcn_last_observation_datetime = None


# ============================================================
# 8. COMMON INPUTS
# ============================================================

st.header("Cyclone Observation")

st.caption(
    "Enter one cyclone observation at a time."
)

# ============================================================
# DATE AND SEASON
# ============================================================

# Row 1: Year, Month, Day
date_col1, date_col2, date_col3 = st.columns(3)

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
    day_options = list(range(1, max_day + 1))
    default_day = min(CURRENT_DAY, max_day)

    # Keep the day valid when month or year changes.
    if st.session_state.get("obs_day") not in day_options:
        st.session_state["obs_day"] = default_day

    day = st.selectbox(
        "Day",
        options=day_options,
        key="obs_day"
    )


# Automatically determine season from the selected month.
season = get_season(month)

SEASON_INFO = {
    1: {
        "name": "Winter",
        "description": (
            "It is a rare period for tropical cyclones to occur due to "
            "generally unfavourable atmospheric conditions."
        )
    },
    2: {
        "name": "Summer",
        "description": (
            "It is the pre-monsoon period that has produced some devastating "
            "cyclones in the past, significantly affecting India."
        )
    },
    3: {
        "name": "Southwest Monsoon (SWM)",
        "description": (
            "It is the monsoon period that has produced relatively more depressions "
            "and deep depressions, many of which do not intensify "
            "into cyclonic storms."
        )
    },
    4: {
        "name": "Northeast Monsoon (NEM)",
        "description": (
            "It is the post-monsoon period which is associated with many "
            "high-intensity cyclones affecting India, particularly "
            "along the east coast."
        )
    }
}

season_name = SEASON_INFO[season]["name"]
season_description = SEASON_INFO[season]["description"]

# Season billboard: displayed separately below the date row.
st.markdown(
    f"""
    <div style="
        border: 2px solid #4682B4;
        border-radius: 12px;
        padding: 16px 24px;
        margin: 8px 0 20px 0;
        background-color: rgba(70, 130, 180, 0.10);
        text-align: center;
    ">
        <div style="
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 1.2px;
            margin-bottom: 6px;
        ">
            SEASON
        </div>
        <div style="
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 6px;
        ">
            {season_name}
        </div>
        <div style="
            font-size: 15px;
            line-height: 1.5;
        ">
            {season_description}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# OBSERVATION TIME — BELOW THE CYCLONE PARAMETERS
# ============================================================

time_col, date_col = st.columns(2)

with time_col:
    hours_display = st.selectbox(
        "Time (UTC)",
        options=[6, 12, 18, 24],
        index=0,
        format_func=lambda x: f"{x:02d}:00",
        key="obs_hours"
    )

    # Convert displayed hours to the model's expected representation.
    hrs_corrected = float(hours_display * 100)

with date_col:
    st.metric(
        "Selected observation date",
        f"{day:02d} {month_display} {year}"
    )

st.caption(
    "The date defaults to the current date. "
    "Change if necessary. Refer IMD bulletin to obtain " 
    "cyclone informations for the selected observation time (UTC)"
)

common_col1, common_col2, common_col3 = st.columns(3)

with common_col1:
    lat = add_input(
        "Latitude (DD)",
        "obs_lat",
        "Latitude of the cyclone centre in decimal degrees.",
        min_value=-90.0,
        max_value=90.0,
        value=15.0,
        format="%.4f"
    )

    lon = add_input(
        "Longitude (DD)",
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
    # Automatically calculate Grade Number from wind speed (knots)
    if 17 <= ws <= 27:
        gradeno = 1
    elif 28 <= ws <= 33:
        gradeno = 2
    elif 34 <= ws <= 47:
        gradeno = 3
    elif 48 <= ws <= 63:
        gradeno = 4
    elif 64 <= ws <= 89:
        gradeno = 5
    elif 90 <= ws <= 119:
        gradeno = 6
    elif ws >= 120:
        gradeno = 7
    else:
        gradeno = 0

    # Display-only category names
    grade_names = {
        0: "Marked Low Pressure",
        1: "Depression (D)",
        2: "Deep Depression (DD)",
        3: "Cyclonic Storm (CS)",
        4: "Severe Syclonic Storm (SCS)",
        5: "Very Severe Cyclonic Storm (VSCS)",
        6: "Extremely Severe Cyclonic Storm (ESCS)",
        7: "Super Cyclonic Storm (SuCS)"
    }

    days_monsoon = add_input(
        "Days After Monsoon Onset",
        "tcn_days_monsoon",
        "Number of days between Kerala southwest monsoon onset "
        "and cyclone genesis. Check IMD official website.",
        value=0.0,
        format="%.4f"
    )

    st.metric("Grade Number", gradeno)
    
st.markdown(
    f"""
    <div style="
        width: fit-content;
        min-width: 320px;
        max-width: 90%;
        box-sizing: border-box;
        background-color: #FEF2F2;
        border: 1px solid #DC2626;
        border-left: 6px solid #DC2626;
        border-radius: 10px;
        padding: 18px 35px;
        margin: 10px auto 16px auto;
        text-align: center;
    ">
        <div style="
            font-size: 16px;
            color: #B91C1C;
            font-weight: 700;
            margin-bottom: 8px;
        ">⚠️ STORM CATEGORY</div>
        <div style="
            font-size: 22px;
            color: #991B1B;
            font-weight: 700;
        ">{grade_names[gradeno]}</div>
    </div>
    """,
    unsafe_allow_html=True
)


# Load the coastline data once and cache it. The dataset must have a valid CRS.
try:
    _coast_mtime = os.path.getmtime(COASTLINE_PATH) if os.path.exists(COASTLINE_PATH) else 0
except OSError:
    _coast_mtime = 0
COASTLINE_GDF, COASTLINE_LOAD_ERROR = load_coastline_data(
    COASTLINE_PATH, _coast_mtime
)
if COASTLINE_LOAD_ERROR:
    st.warning(COASTLINE_LOAD_ERROR)


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
    # ----------------------------------------------------
    # 1. Validate the selected calendar date
    # ----------------------------------------------------
    days_in_month = calendar.monthrange(year, month)[1]

    if not 1 <= day <= days_in_month:
        st.error(
            f"Invalid date: {month_display} {year} has "
            f"{days_in_month} days. Please select a valid date."
        )
        st.stop()

    if COASTLINE_GDF is None:
        st.error(
            "TCN observation was not added because the coastline data "
            "could not be loaded. Fix the coastline path/format and try again."
        )
        st.stop()

    if hours_display not in [6, 12, 18, 24]:
        st.error(
            "Observation time must be 06:00, 12:00, 18:00, or 24:00 UTC."
        )
        st.stop()

    # ----------------------------------------------------
    # 2. Get previous observation timestamps
    # ----------------------------------------------------
    previous_dt = st.session_state.tcn_last_observation_datetime
    first_dt = st.session_state.tcn_first_observation_datetime

    selected_date = datetime(year, month, day)

    # ----------------------------------------------------
    # 3. If 24:00 is selected, require the next calendar date
    # ----------------------------------------------------
    if hours_display == 24 and previous_dt is not None:
        expected_date = previous_dt.date() + timedelta(days=1)
        selected_calendar_date = selected_date.date()

        if selected_calendar_date != expected_date:
            st.error(
                "When selecting 24:00, change the date to the next "
                "calendar day. For example, after 23 Oct 2026 at 18:00, "
                "select 24 Oct 2026 and 24:00."
            )
            st.stop()

    # ----------------------------------------------------
    # 4. Construct the actual timestamp
    #
    # 24:00 is represented as midnight on the following day.
    # The selected date and model input features remain unchanged.
    # ----------------------------------------------------
    if hours_display == 24:
        observation_dt = selected_date + timedelta(days=1)
    else:
        observation_dt = selected_date.replace(hour=hours_display)

    # ----------------------------------------------------
    # 5. Validate chronological order and exact 6-hour interval
    # ----------------------------------------------------
    if previous_dt is not None:
        elapsed_hours = (
            observation_dt - previous_dt
        ).total_seconds() / 3600.0

        if elapsed_hours <= 0:
            st.error(
                "Observation rejected: the selected timestamp must be "
                "later than the previous observation."
            )
            st.stop()

        if elapsed_hours != 6:
            st.error(
                f"Observation rejected: the interval from the previous "
                f"observation ({previous_dt.strftime('%d %b %Y, %H:%M')} UTC) "
                f"to the selected observation "
                f"({observation_dt.strftime('%d %b %Y, %H:%M')} UTC) "
                f"is {elapsed_hours:g} hours. Exactly 6 hours is required."
            )
            st.stop()

    # ----------------------------------------------------
    # 6. Calculate automated TCN features
    # ----------------------------------------------------
    try:
        # Direction from the previous observed location
        if st.session_state.tcn_track_history:
            previous_observation = st.session_state.tcn_track_history[-1]

            direction = calculate_bearing(
                float(previous_observation["Lat"]),
                float(previous_observation["Lon"]),
                float(lat),
                float(lon),
            )
        else:
            direction = 0.0

        # Duration in hours since the first observation
        duration = (
            0.0 if first_dt is None
            else (
                observation_dt - first_dt
            ).total_seconds() / 3600.0
        )

        # Find nearest coastline point and distance
        nearest_lat, nearest_lon, distance_coast = find_nearest_coast(
            float(lat),
            float(lon),
            COASTLINE_GDF
        )

        # ----------------------------------------------------
        # 7. Prepare TCN input
        #
        # Preserve the selected training convention:
        # Day = selected calendar day
        # hrs_corrected = 2400 when 24:00 is selected
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
        # 8. Prepare LSTM input
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

        # ----------------------------------------------------
        # 9. Save observations only after validation succeeds
        # ----------------------------------------------------
        if first_dt is None:
            st.session_state.tcn_first_observation_datetime = observation_dt

        st.session_state.tcn_last_observation_datetime = observation_dt

        st.session_state.tcn_track_history.append(tcn_observation)
        st.session_state.lstm_track_history.append(lstm_observation)

        st.success(
            f"Observation added successfully: "
            f"{day:02d} {month_display} {year}, "
            f"{hours_display:02d}:00 UTC."
        )

        st.caption(
            f"Actual timestamp used for interval calculations: "
            f"{observation_dt.strftime('%d %b %Y, %H:%M')} UTC."
        )

        st.caption(
            f"Automated TCN features — direction: {direction:.2f}°, "
            f"duration: {duration:.2f} h, nearest coast: "
            f"({nearest_lat:.4f}°, {nearest_lon:.4f}°), "
            f"distance: {distance_coast:.2f} km."
        )

    except Exception as exc:
        st.error(
            f"Could not calculate automated TCN inputs: {exc}"
        )

# ============================================================
# 12. DISPLAY SHARED OBSERVATION HISTORY
# ============================================================

st.divider()
st.subheader("Observation History")

# Use one history for display to avoid duplicate observations.
# LSTM history contains all eight shared observation fields.
lstm_history = st.session_state.lstm_track_history

if lstm_history:
    observation_df = pd.DataFrame(lstm_history)

    # Convert the stored model time (e.g. 600.0) to HH:MM.
    def format_observation_time(value):
        if pd.isna(value):
            return None

        value = int(value)
        hours = value // 100
        minutes = value % 100
        return f"{hours:02d}:{minutes:02d}"

    if "hrs_corrected" in observation_df.columns:
        observation_df["Time"] = (
            observation_df["hrs_corrected"]
            .apply(format_observation_time)
        )

    display_columns = [
        "Day", "Month", "Year", "Time",
        "Lat", "Lon", "WS", "CP"
    ]

    # Ensure the table always has the requested columns.
    for col in display_columns:
        if col not in observation_df.columns:
            observation_df[col] = None

    observation_df = observation_df[display_columns]

    observation_df = observation_df.sort_values(
        by=["Year", "Month", "Day", "Time"],
        kind="stable"
    ).reset_index(drop=True)

    # Read-only table; do not use st.data_editor here.
    st.dataframe(
        observation_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No observations have been added yet.")

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
        st.session_state.tcn_first_observation_datetime = None
        st.session_state.tcn_last_observation_datetime = None
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