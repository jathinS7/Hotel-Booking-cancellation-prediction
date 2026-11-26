# Hotel Booking Cancellation Prediction

Predicts whether a hotel booking will be cancelled, using a Random Forest classifier served through a FastAPI endpoint with a Streamlit front end.

Dataset: 119,390 hotel bookings from two Portuguese hotels (City and Resort), 2015–2017. Target: `is_canceled`.

## Method

1. **Clean:** drop exact duplicate rows (about 27% of the data). Check missing values per column.
2. **Avoid leakage:** drop `reservation_status` and `reservation_status_date`, which are only known after the stay ends.
3. **Split by time:** order bookings by booking date (arrival date minus lead time) and split 70 / 15 / 15 into train, validation and test. The test set is the most recent bookings, which matches how the model would be used in practice.
4. **Model:** median imputation and scaling for numbers, most-frequent imputation and one-hot encoding for categories, then a Random Forest with balanced class weights.
5. **Threshold:** pick the decision threshold that maximises F1 on the validation set, not a fixed 0.5.
6. **Final fit:** refit on train + validation with the same settings, then save the model and threshold.

## Results

Results from the notebook on the full dataset. Test set: 13,110 bookings. Final model (trained on train + validation):

| Metric | Value |
|---|---|
| Accuracy | 0.785 |
| Precision (cancelled) | 0.62 |
| Recall (cancelled) | 0.54 |
| F1 (cancelled) | 0.58 |
| ROC-AUC | 0.834 |
| PR-AUC | 0.633 |

The notebook also tried logistic regression and a decision tree on the validation set. The forest was the strongest of the three. The notebook's `GridSearchCV` search (trees, depth and leaf size, with `TimeSeriesSplit`) is in the notebook.

> The numbers above come from the notebook. `src/train.py` uses the same method but fixed hyperparameters (300 trees, depth 12, leaf size 3), not the notebook's grid-search winner. Its results on the real data are not quoted here. Run `python -m src.train` on the real CSV to get them.

## Project structure

```
data/           hotel_bookings.csv goes here (not committed)
models/         saved model and threshold (created by training)
notebooks/      original exploratory analysis and experiments
outputs/        metrics.json (created by training)
src/
  config.py     paths and constants
  data.py       dedupe, booking date, chronological split
  model.py      feature selection and sklearn pipeline
  train.py      training, threshold selection, evaluation
  predict.py    loads the saved model and scores one booking
  api.py        FastAPI app
  ui.py         Streamlit dashboard
tests/          unit tests for data handling and the API
```

## Setup

```bash
pip install -r requirements.txt
```

Download `hotel_bookings.csv` (the Hotel Booking Demand dataset) and put it in `data/`.

## Train

```bash
python -m src.train
```

This writes `models/hotel_cancellation_model.pkl`, `models/decision_threshold.json` and `outputs/metrics.json`.

## Run the API and dashboard

```bash
uvicorn src.api:app --port 8000
# in a second terminal
streamlit run src/ui.py
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"hotel":"City Hotel","lead_time":200,"arrival_date_year":2017,
       "arrival_date_month":"July","arrival_date_week_number":25,
       "arrival_date_day_of_month":1,"stays_in_weekend_nights":1,
       "stays_in_week_nights":2,"adults":2,"children":0,"babies":0,
       "meal":"BB","country":"PRT","market_segment":"Online TA",
       "distribution_channel":"TA/TO","is_repeated_guest":0,
       "previous_cancellations":0,"previous_bookings_not_canceled":0,
       "reserved_room_type":"A","agent":9,"days_in_waiting_list":0,
       "customer_type":"Transient","adr":95.5,
       "required_car_parking_spaces":0,"total_of_special_requests":0,
       "deposit_type":"No Deposit"}'
```

Response shape (values depend on the trained model):

```json
{"cancellation_probability": 0.0, "threshold": 0.0,
 "prediction": 0, "action_recommended": "No action required"}
```

## Tests

```bash
pytest
```

The tests cover the booking-date calculation, the chronological split, the leakage exclusions, and API input validation.

## Limitations

- The model predicts cancellation, but it does not explain why. Feature importance is not yet reported.
- The dashboard fills in fixed defaults (country, meal, room type, agent) for fields it does not ask for. Those defaults affect the prediction.
- The dataset is from 2015–2017 and two hotels, so results may not transfer to other periods or properties.

## Author

Chillagattu Jathin · [LinkedIn](https://www.linkedin.com/in/jathins7) · [Portfolio](https://jathindsprojects.netlify.app/)
