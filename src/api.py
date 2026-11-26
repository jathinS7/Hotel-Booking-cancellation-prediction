"""FastAPI service: POST /predict returns the cancellation risk for one booking."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_booking

app = FastAPI(
    title="Hotel Cancellation Prediction API",
    description="Predicts whether a hotel booking will be cancelled.",
    version="1.1.0",
)


class BookingInput(BaseModel):
    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: float = Field(ge=0)
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: float = Field(ge=0)
    babies: int = Field(ge=0)
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int = Field(ge=0, le=1)
    previous_cancellations: int = Field(ge=0)
    previous_bookings_not_canceled: int = Field(ge=0)
    reserved_room_type: str
    agent: float
    days_in_waiting_list: int = Field(ge=0)
    customer_type: str
    adr: float = Field(ge=0)
    required_car_parking_spaces: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)
    deposit_type: str


@app.get("/")
def home():
    return {"message": "Hotel cancellation API is running."}


@app.post("/predict")
def predict(booking: BookingInput):
    return predict_booking(booking.model_dump())
