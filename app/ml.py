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

model_high_location = r'models/catboost 75.sav'
model_medium_location = r'models/catboost_50.sav'
model_low_location = r'models/catboost_25.sav'

model_high = pickle.load(open(model_high_location, 'rb'))
model_medium = pickle.load(open(model_medium_location, 'rb'))
model_low = pickle.load(open(model_low_location, 'rb'))

class RentalUnit(BaseModel):
    """Use this data model to parse the request body JSON."""

    borough: str = 'Queens'
    room_type: str = 'room'
    accommodates: int = 4
    day_of_week: int = 6
    days_until_booking: int = 7


    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([self.dict()])

    @validator('days_until_booking')
    def days_until_positive(cls, value):
        """Validate that x1 is a positive number."""
        assert value > 0, f'days until booking == {value}, must be > 0'
        return value


@router.post('/predict')
async def predict(rentalUnit: RentalUnit):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `borough`: str
    - `room_type`: str
    - `accommodates`: int
    - 'day_of_week' : int
    - 'days_until_booking' : int

    ### Response
    - `prediction_high`: estimate of 75th percentile price
    - 'prediction_medium': estimate of mean price - not percentile
    - 'prediction_low': estimate of 25th percentile price  

    
    """
    rentalunit = RentalUnit()
    rentdict = dict(rentalunit)
    
    df = pd.DataFrame(jsonable_encoder(rentalunit), index = [0])
    log.info(df)
    high_pred = (model_high.predict(df)).astype(int)
    medium_pred = (model_medium.predict(df)).astype(int)
    low_pred = (model_low.predict(df)).astype(int)
    
    return {
        'prediction high': str(high_pred[0]),
        'prediction medium': str(medium_pred[0]),
        'prediction low': str(low_pred[0])
    }
