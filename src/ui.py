"""Streamlit front end for the cancellation API.

Run the API first (uvicorn src.api:app --port 8000), then: streamlit run src/ui.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("HOTEL_API_URL", "http://127.0.0.1:8000/predict")
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

st.set_page_config(page_title="Hotel Booking Cancellation Predictor")
st.title("Hotel booking cancellation predictor")
st.write("Enter the booking details to estimate the cancellation risk.")

col1, col2, col3 = st.columns(3)
with col1:
    hotel = st.selectbox("Hotel type", ["City Hotel", "Resort Hotel"])
    lead_time = st.number_input("Lead time (days)", min_value=0, value=30)
    market_segment = st.selectbox(
        "Market segment",
        ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Complementary", "Aviation"],
    )
    deposit_type = st.selectbox("Deposit type", ["No Deposit", "Non Refundable", "Refundable"])
with col2:
    arrival_month = st.selectbox("Arrival month", MONTHS, index=6)
    arrival_day = st.number_input("Arrival day of month", min_value=1, max_value=31, value=1)
    adr = st.number_input("Average daily rate", min_value=0.0, value=95.5)
    special_requests = st.number_input("Special requests", min_value=0, value=0)
with col3:
    customer_type = st.selectbox("Customer type", ["Transient", "Transient-Party", "Contract", "Group"])
    distribution = st.selectbox("Distribution channel", ["Direct", "TA/TO", "Corporate", "GDS"])
    parking = st.number_input("Required car parking spaces", min_value=0, value=0)
    prev_cancellations = st.number_input("Previous cancellations", min_value=0, value=0)

if st.button("Analyse risk"):
    payload = {
        "hotel": hotel,
        "lead_time": float(lead_time),
        "arrival_date_year": 2017,
        "arrival_date_month": arrival_month,
        "arrival_date_week_number": 25,
        "arrival_date_day_of_month": int(arrival_day),
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0.0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": market_segment,
        "distribution_channel": distribution,
        "is_repeated_guest": 0,
        "previous_cancellations": int(prev_cancellations),
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "agent": 9.0,
        "days_in_waiting_list": 0,
        "customer_type": customer_type,
        "adr": float(adr),
        "required_car_parking_spaces": int(parking),
        "total_of_special_requests": int(special_requests),
        "deposit_type": deposit_type,
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException:
        st.warning(f"Could not reach the API at {API_URL}. Start it first.")
    else:
        st.subheader("Result")
        st.metric("Cancellation probability", f"{result['cancellation_probability'] * 100:.1f}%")
        if result["prediction"] == 1:
            st.error(f"High risk: {result['action_recommended']}.")
        else:
            st.success("Low risk: no action required.")
