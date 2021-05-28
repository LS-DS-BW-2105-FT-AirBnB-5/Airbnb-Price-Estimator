"""Machine learning functions"""

import logging
import random
import sklearn
import catboost
import category_encoders
import pickle

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
from fastapi.encoders import jsonable_encoder

log = logging.getLogger(__name__)
router = APIRouter()

model_high_location = r'models/catboost_75c.sav'
model_medium_location = r'models/catboost_50c.sav'
model_low_location = r'models/catboost_25c.sav'

model_high = pickle.load(open(model_high_location, 'rb'))
model_medium = pickle.load(open(model_medium_location, 'rb'))
model_low = pickle.load(open(model_low_location, 'rb'))

class RentalUnit(BaseModel):
    """A model to parse the request body JSON."""

    borough: str = Field(..., example='Queens')
    room_type: str = Field(..., example='room')
    accommodates: int = Field(..., example=4)
    day_of_week: int = Field(..., example=6)
    days_until_booking: int = Field(..., example=7)


    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame(jsonable_encoder(rentalUnit), index = [0])

    @validator('days_until_booking')
    def days_until_positive(cls, value):
        """Validate that days until booking is a positive number."""
        assert value > 0, f'days until booking == {value}, must be > 0'
        return value


@router.post('/predict')
async def predict(rentalUnit: RentalUnit):
    """
    Predicting a high, medium, and low price for a rental unit 🔮

    ### Request Body
    - `borough`: str
    - `room_type`: str
    - `accommodates`: int
    - 'day_of_week' : int
    - 'days_until_booking' : int

    ### Response
    - `prediction_high`: estimate of 75th percentile price
    - 'prediction_medium': estimate of median (50th) price
    - 'prediction_low': estimate of 25th percentile price  

    
    """
    df = pd.DataFrame(jsonable_encoder(rentalUnit), index = [0])
    log.info(df)
    high_pred = (model_high.predict(df)).astype(int)
    medium_pred = (model_medium.predict(df)).astype(int)
    low_pred = (model_low.predict(df)).astype(int)
    
    return {
        'prediction high': str(high_pred[0]),
        'prediction medium': str(medium_pred[0]),
        'prediction low': str(low_pred[0])
    }
