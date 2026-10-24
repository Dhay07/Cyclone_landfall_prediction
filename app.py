import streamlit as st
import pandas as pd
import json
import os
import calendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math
# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cyclone Landfall Prediction-BoB",
    page_icon="🌀",
    layout="wide"
)

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)
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
from google_sheets_api import send_to_google_sheets

if st.sidebar.button("Test Google Sheets connection"):
    try:
        result = send_to_google_sheets("ping")
        st.sidebar.success(result["message"])

    except Exception as e:
        st.sidebar.error(f"Error type: {type(e).__name__}")
        st.sidebar.code(str(e))

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
    - The models are primarily intended for the Bay of Bengal;
      performance may vary across cyclones and seasons.

    Current cyclone observations may be obtained from
    [India Meteorological Department](https://mausam.imd.gov.in/)
    and the [Windy](https://www.windy.com/).

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

st.markdown(
    """
    <h1 style="text-align: center;">
        🌀 Cyclone Landfall Prediction - Bay of Bengal
    </h1>
    <p style="text-align: center; font-size: 16px;">
        Research-based tropical cyclone prediction using
        deep-learning models.
    </p>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 7. INITIALIZE SEPARATE MODEL HISTORIES
# ============================================================

if "tcn_track_history" not in st.session_state:
    st.session_state.tcn_track_history = []

if "lstm_track_history" not in st.session_state:
    st.session_state.lstm_track_history = []
if "lstm_predicted_lat" not in st.session_state:
    st.session_state.lstm_predicted_lat = None

if "lstm_predicted_lon" not in st.session_state:
    st.session_state.lstm_predicted_lon = None

if "tcn_predicted_lat" not in st.session_state:
    st.session_state.tcn_predicted_lat = None

if "tcn_predicted_lon" not in st.session_state:
    st.session_state.tcn_predicted_lon = None

# Timestamps are stored separately so no helper columns are added to the
# DataFrame passed into the trained TCN model.
if "tcn_first_observation_datetime" not in st.session_state:
    st.session_state.tcn_first_observation_datetime = None
if "tcn_last_observation_datetime" not in st.session_state:
    st.session_state.tcn_last_observation_datetime = None

left_space, content_col, right_space = st.columns([0.05, 0.90, 0.05])

with content_col:

    st.header("Cyclone Observation")
    st.caption("Enter one cyclone observation at a time.")

    # ============================================================
    # TWO-COLUMN LAYOUT
    # LEFT: DATE AND OBSERVATION INPUTS
    # RIGHT: SEASON AND CYCLONE CATEGORY BILLBOARDS
    # ============================================================

    input_col, billboard_col = st.columns(
        [2.1, 1],
        gap="large"
    )

    # ============================================================
    # LEFT COLUMN: DATE, TIME AND OBSERVATION INPUTS
    # ============================================================

    with input_col:

        st.subheader("Observation Date and Time")

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

            if st.session_state.get("obs_day") not in day_options:
                st.session_state["obs_day"] = default_day

            day = st.selectbox(
                "Day",
                options=day_options,
                key="obs_day"
            )

        # Determine season automatically from the selected month.
        season = get_season(month)

        # Observation time
        time_col, selected_date_col = st.columns(2)

        with time_col:
            hours_display = st.selectbox(
                "Time (UTC)",
                options=[0, 6, 12, 18],
                index=1,
                format_func=lambda x: f"{x:02d}:00",
                key="obs_hours"
            )

            hrs_corrected = float(
                2400 if hours_display == 0
                else hours_display * 100
            )

        with selected_date_col:
            st.metric(
                "Selected observation date",
                f"{day:02d} {month_display} {year}"
            )
            st.caption(f"Observation time: {hours_display:02d}:00 UTC")

        st.divider()

        st.subheader("Cyclone Parameters")

        common_col1, common_col2 = st.columns(2)

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

            ws = add_input(
                "Wind Speed (knots)",
                "obs_ws",
                "Maximum sustained wind speed at the observation time, in knots.",
                min_value=0.0,
                value=40.0,
                format="%.4f"
            )

        with common_col2:
            cp = add_input(
                "Central Pressure (hPa)",
                "obs_cp",
                "Estimated minimum central pressure at the observation time, in hectopascals.",
                min_value=0.0,
                value=1000.0,
                format="%.4f"
            )

            days_monsoon = add_input(
                "Days After Monsoon Onset",
                "tcn_days_monsoon",
                "Number of days between Kerala southwest monsoon onset and cyclone genesis. Check the official IMD website.",
                value=0.0,
                format="%.4f"
            )

        # Calculate cyclone grade from wind speed.
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

        grade_names = {
            0: "Marked Low Pressure",
            1: "Depression (D)",
            2: "Deep Depression (DD)",
            3: "Cyclonic Storm (CS)",
            4: "Severe Cyclonic Storm (SCS)",
            5: "Very Severe Cyclonic Storm (VSCS)",
            6: "Extremely Severe Cyclonic Storm (ESCS)",
            7: "Super Cyclonic Storm (SuCS)"
        }

    # ============================================================
    # 12. DISPLAY SHARED OBSERVATION HISTORY
    # ============================================================
        # ============================================================
        # OBSERVATION HISTORY
        # ============================================================
        st.divider()
        st.subheader("Observation History")

        # Use LSTM history to avoid displaying duplicate observations.
        lstm_history = st.session_state.lstm_track_history

        if lstm_history is not None and len(lstm_history) > 0:
            observation_df = pd.DataFrame(lstm_history).copy()

            # Convert stored model time (e.g., 600.0) to HH:MM.
            def format_observation_time(value):
                if pd.isna(value):
                    return None

                value = int(value)

                if value == 2400:
                    return "00:00"

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

            # Ensure all requested columns exist.
            for col in display_columns:
                if col not in observation_df.columns:
                    observation_df[col] = None

            observation_df = observation_df[display_columns]

            # Sort observations chronologically.
            observation_df = observation_df.sort_values(
                by=["Year", "Month", "Day", "Time"],
                kind="stable",
                na_position="last"
            ).reset_index(drop=True)

            # Display a read-only history table.
            st.dataframe(
                observation_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No observations have been added yet.")

    # ============================================================
    # RIGHT COLUMN: SEASON AND STORM CATEGORY BILLBOARDS
    # ============================================================

    with billboard_col:
        
        SEASON_INFO = {
            1: {
                "name": "Winter",
                "description": (
                    "A rare period for tropical cyclones due to "
                    "generally unfavourable atmospheric conditions."
                )
            },
            2: {
                "name": "Summer",
                "description": (
                    "The pre-monsoon period, which has produced "
                    "some devastating cyclones affecting India."
                )
            },
            3: {
                "name": "Southwest Monsoon (SWM)",
                "description": (
                    "This period produces depressions and deep "
                    "depressions, many of which do not intensify "
                    "into cyclonic storms."
                )
            },
            4: {
                "name": "Northeast Monsoon (NEM)",
                "description": (
                    "The post-monsoon period associated with many "
                    "high-intensity cyclones affecting India's east coast."
                )
            }
        }

        season_name = SEASON_INFO[season]["name"]
        season_description = SEASON_INFO[season]["description"]

        st.markdown(
            f"""
            <div style="
                border: 2px solid #4682B4;
                border-radius: 12px;
                padding: 20px 16px;
                margin: 0 0 20px 0;
                background-color: rgba(70, 130, 180, 0.10);
                text-align: center;
                min-height: 185px;
                box-sizing: border-box;
            ">
                <div style="
                    font-size: 12px;
                    font-weight: 600;
                    letter-spacing: 1.2px;
                    margin-bottom: 10px;
                ">SEASON</div>
                <div style="
                    font-size: 23px;
                    font-weight: 700;
                    margin-bottom: 12px;
                ">{season_name}</div>
                <div style="
                    font-size: 14px;
                    line-height: 1.5;
                ">{season_description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        category_colors = {
            0: ("#64748B", "rgba(100, 116, 139, 0.10)"),
            1: ("#0D9488", "rgba(13, 148, 136, 0.10)"),
            2: ("#0891B2", "rgba(8, 145, 178, 0.10)"),
            3: ("#2563EB", "rgba(37, 99, 235, 0.10)"),
            4: ("#D97706", "rgba(217, 119, 6, 0.10)"),
            5: ("#EA580C", "rgba(234, 88, 12, 0.10)"),
            6: ("#DC2626", "rgba(220, 38, 38, 0.10)"),
            7: ("#9333EA", "rgba(147, 51, 234, 0.10)")
        }

        category_color, category_bg = category_colors[gradeno]

        st.markdown(
            f"""
            <div style="
                border: 2px solid {category_color};
                border-left: 7px solid {category_color};
                border-radius: 12px;
                padding: 24px 16px;
                margin: 0 0 16px 0;
                background-color: {category_bg};
                text-align: center;
                min-height: 150px;
                box-sizing: border-box;
            ">
                <div style="
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 1.1px;
                    margin-bottom: 14px;
                ">STORM CATEGORY</div>
                <div style="
                    font-size: 21px;
                    font-weight: 700;
                    line-height: 1.4;
                ">{grade_names[gradeno]}</div>
                <div style="
                    font-size: 14px;
                    margin-top: 12px;
                ">Grade Number: {gradeno}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Season updates automatically with the selected month. "
            "Storm category updates automatically with wind speed."
        )
        prediction_panel = st.container()
    
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
    # 11. OBSERVATION, PREDICT AND RESET BUTTONS
    # ============================================================

    st.divider()

    add_col, predict_col, reset_col = st.columns([2, 1, 1])

    with add_col:
        submitted = st.button(
            "Add Observation",
            type="primary",
            use_container_width=True
        )

    with predict_col:
        predict_clicked = st.button(
            "Predict",
            use_container_width=True
        )

    with reset_col:
        reset_clicked = st.button(
            "Reset",
            use_container_width=True
        )

    # Keep prediction results hidden until the user clicks Predict.
    if "show_predictions" not in st.session_state:
        st.session_state.show_predictions = False

    # Reset both model histories and all associated timestamps.
    if reset_clicked:
        st.session_state.tcn_track_history = []
        st.session_state.lstm_track_history = []

        st.session_state.tcn_first_observation_datetime = None
        st.session_state.tcn_last_observation_datetime = None

    # Clear prediction values.
        st.session_state.lstm_predicted_lat = None
        st.session_state.lstm_predicted_lon = None
        st.session_state.tcn_predicted_lat = None
        st.session_state.tcn_predicted_lon = None

        st.session_state.show_predictions = False
        st.rerun()

    # A prediction is requested only when the Predict button is clicked.
    if predict_clicked:
        lstm_min_observations = int(LSTM_CONFIG.get("window_size", 2))
        tcn_min_observations = 2

        if len(st.session_state.lstm_track_history) < lstm_min_observations:
            st.session_state.show_predictions = False
            st.warning(
                f"Add at least {lstm_min_observations} observations before "
                "generating track prediction."
            )
        elif len(st.session_state.tcn_track_history) < tcn_min_observations:
            st.session_state.show_predictions = False
            st.warning(
                f"Add at least {tcn_min_observations} observations before "
                "generating landfall prediction."
            )

        else:
            try:
                lstm_track_df = pd.DataFrame(
                    st.session_state.lstm_track_history
                )

                tcn_track_df = pd.DataFrame(
                    st.session_state.tcn_track_history
                )

                lstm_result = predict_lstm(lstm_track_df)
                tcn_result = predict_tcn(tcn_track_df)

                st.session_state.lstm_predicted_lat = (
                    lstm_result["predicted_lat"]
                )
                st.session_state.lstm_predicted_lon = (
                    lstm_result["predicted_lon"]
                )

                st.session_state.tcn_predicted_lat = (
                    tcn_result["predicted_landfall_lat"]
                )
                st.session_state.tcn_predicted_lon = (
                    tcn_result["predicted_landfall_lon"]
                )

                st.session_state.show_predictions = True

            except Exception as e:
                st.session_state.show_predictions = False
                st.error(f"Prediction failed: {e}")

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

        if hours_display not in [0, 6, 12, 18]:
            st.error(
                "Observation time must be 00:00, 06:00, 12:00, or 18:00 UTC."
            )
            st.stop()

        # ----------------------------------------------------
        # 2. Get previous observation timestamps
        # ----------------------------------------------------
        previous_dt = st.session_state.tcn_last_observation_datetime
        first_dt = st.session_state.tcn_first_observation_datetime

        # ----------------------------------------------------
        # 3. Construct the actual timestamp
        # 00:00 means midnight on the selected calendar date
        # ----------------------------------------------------
        selected_date = datetime(year, month, day)
        observation_dt = selected_date.replace(hour=hours_display)

        # ----------------------------------------------------
        # 4. Validate chronological order and exact 6-hour interval
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
            # hrs_corrected = 2400 for the displayed 00:00 time.
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


            st.session_state.lstm_predicted_lat = None
            st.session_state.lstm_predicted_lon = None
            st.session_state.tcn_predicted_lat = None
            st.session_state.tcn_predicted_lon = None

            st.session_state.show_predictions = False

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
            st.rerun()

        except Exception as exc:
            st.error(
                f"Could not calculate automated TCN inputs: {exc}"
            )


# ============================================================
# MODEL PREDICTIONS — DISPLAY BELOW THE RIGHT-SIDE BILLBOARDS
# ============================================================

with prediction_panel:

    # ========================================================
    # LSTM TRACK PREDICTION
    # ========================================================

    st.divider()
    st.subheader("Predicted Track 12 h Ahead")

    st.caption(
        f"Model version: {LSTM_CONFIG['model_version']}"
    )

    lat_col, lon_col = st.columns(2)

    with lat_col:
        st.metric(
            "Predicted Latitude (DD)",
            (
                f"{st.session_state.lstm_predicted_lat:.4f}"
                if st.session_state.lstm_predicted_lat is not None
                else "—"
            )
        )

    with lon_col:
        st.metric(
            "Predicted Longitude (DD)",
            (
                f"{st.session_state.lstm_predicted_lon:.4f}"
                if st.session_state.lstm_predicted_lon is not None
                else "—"
            )
        )

    # ========================================================
    # TCN LANDFALL PREDICTION
    # ========================================================

    st.divider()
    st.subheader("Predicted Landfall")

    st.caption(
        f"Model version: {TCN_CONFIG['model_version']}"
    )

    lat_col, lon_col = st.columns(2)

    with lat_col:
        st.metric(
            "Landfall Latitude (DD)",
            (
                f"{st.session_state.tcn_predicted_lat:.4f}"
                if st.session_state.tcn_predicted_lat is not None
                else "—"
            )
        )

    with lon_col:
        st.metric(
            "Landfall Longitude (DD)",
            (
                f"{st.session_state.tcn_predicted_lon:.4f}"
                if st.session_state.tcn_predicted_lon is not None
                else "—"
            )
        )

# ============================================================
# 16. FOOTER: ABOUT, CONTACT, EXTERNAL LINKS AND DISCLAIMER
# ============================================================

st.divider()

center_col, contact_col, links_col = st.columns(3)

with center_col:
    if st.button("ℹ️ About", use_container_width=True, type="secondary"):
        show_about()

with contact_col:
    with st.expander("✉️ Contact"):
        st.markdown("""
         For questions, feedback, or research collaboration, please feel
         free to get in touch.

        - **Developer:** Dhayanand J
        - **Email:** dhayanand06@gmail.com
        - **Institution:** SASTRA Deemed University
        """)

with links_col:
    with st.expander("🔗 External Links"):
        st.markdown("""
        - [India Meteorological Department (IMD)](https://mausam.imd.gov.in/)
        - [RSMC New Delhi](https://rsmcnewdelhi.imd.gov.in/)
        """)

st.divider()

st.caption(
    "Disclaimer: This application presents predictions from "
    "research-based deep-learning models. It is not an official "
    "cyclone warning or forecasting service. Predictions must not "
    "be used as a substitute for official forecasts, advisories, "
    "or warnings issued by the IMD or other authorized agencies."
)

st.caption(
    "© 2026 Cyclone Landfall Prediction | Research and educational use"
)