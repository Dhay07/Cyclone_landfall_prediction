import hashlib
import hmac
import secrets
import time
from datetime import datetime, timezone, timedelta

import streamlit as st
import extra_streamlit_components as stx


COOKIE_NAME = "cyclone_visitor_token"
COOKIE_LIFETIME_DAYS = 365


def _sign(payload: str) -> str:
    secret = st.secrets["visitor_identity"]["signing_secret"]

    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()


def _verify_cookie(cookie_value: str):
    """Return the visitor ID if the signed cookie is valid."""

    try:
        token, expiry_text, supplied_signature = cookie_value.rsplit(".", 2)
        payload = f"{token}.{expiry_text}"

        if not token or int(expiry_text) <= int(time.time()):
            return None

        expected_signature = _sign(payload)

        if not hmac.compare_digest(
            supplied_signature,
            expected_signature
        ):
            return None

        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    except (ValueError, TypeError):
        return None


def get_visitor_identity(cookie_manager):
    """
    Return a stable visitor ID from a signed browser cookie.
    Create a new identity if no valid cookie exists.
    """

    # Reuse identity during Streamlit reruns in this session.
    cached_id = st.session_state.get("_verified_visitor_id")

    if cached_id:
        return cached_id

    cookie_value = cookie_manager.get(COOKIE_NAME)

    if cookie_value:
        visitor_id = _verify_cookie(cookie_value)

        if visitor_id:
            st.session_state["_verified_visitor_id"] = visitor_id
            return visitor_id

    # Generate a new anonymous visitor token.
    token = secrets.token_urlsafe(32)
    expiry = int(time.time()) + (
        COOKIE_LIFETIME_DAYS * 24 * 60 * 60
    )

    payload = f"{token}.{expiry}"
    signature = _sign(payload)
    cookie_value = f"{payload}.{signature}"

    expires_at = datetime.fromtimestamp(
        expiry,
        tz=timezone.utc
    )

    cookie_manager.set(
        COOKIE_NAME,
        cookie_value,
        expires_at=expires_at,
        key="set_cyclone_visitor_cookie"
    )

    visitor_id = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    st.session_state["_verified_visitor_id"] = visitor_id

    return visitor_id
