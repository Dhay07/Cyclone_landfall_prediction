
import requests
import streamlit as st


def send_to_google_sheets(action, data=None):
    """Send a server-side request to the Google Apps Script API."""

    config = st.secrets["google_sheets"]

    url = config["apps_script_url"]
    api_key = config["api_key"]

    payload = {
        "api_key": api_key,
        "action": action,
        "data": data or {}
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(
            result.get("error", "Google Sheets API request failed.")
        )

    return result
