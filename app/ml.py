"""Machine learning functions"""

import logging
import random
import sklearn
import xgboost
import catboost
import category_encoders
import pickle

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
from fastapi.encoders import jsonable_encoder

log = logging.getLogger(__name__)
router = APIRouter()

model_location = r'models\catboost 75.sav'
model_locationb = r'models\catboost_50.sav'
model_locationc = r'models\catboost_25.sav'

model = pickle.load(open(model_location, 'rb'))
modelb = pickle.load(open(model_locationb, 'rb'))
modelc = pickle.load(open(model_locationc, 'rb'))

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
async def predict():
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `borough`: str
    - `room_type`: str
    - `accommodates`: int
    - 'day_of_week' : int
    - 'days_until_booking' : int

    ### Response
    - `prediction`: estimate of expected price   

    
    """
    rentalunit = RentalUnit()
    rentdict = dict(rentalunit)
    df = pd.DataFrame(jsonable_encoder(rentalunit), index = [0])
    # # rentdf = rentalunit.to_df()
    # rentdf = pd.DataFrame(rentdict, index = [0])
    # # # log.info(RU_df)
    y_pred = (model.predict(df)).astype(int)
    y_predb = (modelb.predict(df)).astype(int)
    y_predc = (modelc.predict(df)).astype(int)
    return {
        'prediction high': str(y_pred[0]),
        'prediction medium': str(y_predb[0]),
        'prediction low': str(y_predc[0])
    }
