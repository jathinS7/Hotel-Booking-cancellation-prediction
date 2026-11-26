from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

VALID = {
    "hotel": "City Hotel", "lead_time": 30.0, "arrival_date_year": 2017,
    "arrival_date_month": "July", "arrival_date_week_number": 25,
    "arrival_date_day_of_month": 1, "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2, "adults": 2, "children": 0.0, "babies": 0,
    "meal": "BB", "country": "PRT", "market_segment": "Online TA",
    "distribution_channel": "TA/TO", "is_repeated_guest": 0,
    "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A", "agent": 9.0, "days_in_waiting_list": 0,
    "customer_type": "Transient", "adr": 95.5, "required_car_parking_spaces": 0,
    "total_of_special_requests": 0, "deposit_type": "No Deposit",
}


def test_home_responds():
    assert client.get("/").status_code == 200


def test_rejects_invalid_hotel():
    bad = {**VALID, "hotel": "Mars Hotel"}
    assert client.post("/predict", json=bad).status_code == 422


def test_rejects_negative_lead_time():
    bad = {**VALID, "lead_time": -1}
    assert client.post("/predict", json=bad).status_code == 422
