from fastapi import FastAPI 
from pydantic import BaseModel , Field
import joblib 
import json
import pandas as pd
from typing import Literal
app = FastAPI()
model = joblib.load("house_price_model.pkl")
with open('model_columns.json','r') as f:
    model_columns = json.load(f)
@app.get("/")
def home():
    return{"message":"House Price Prediction is running"}
class HouseFeatures(BaseModel):
    area : int = Field(...,gt=0)
    bedrooms : int = Field(...,ge=1,le=6)
    bathrooms: int = Field(...,ge=1,le=4)
    stories : int = Field(...,ge=1,le=4)
    mainroad:Literal["yes","no"]
    guestroom:Literal["yes","no"]
    basement : Literal["yes","no"]
    hotwaterheating : Literal["yes","no"]
    airconditioning :Literal["yes","no"]
    parking: int = Field(..., ge=0, le=3)
    prefarea: Literal["yes", "no"]
    furnishingstatus: Literal["furnished", "semi-furnished", "unfurnished"]
@app.post("/predict")
def predict_price(features : HouseFeatures):
    input_dict = features.model_dump()

    binary_cols= ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]

    for col in binary_cols:
        input_dict[col] = 1 if input_dict[col] == "yes" else 0

    furnishing = input_dict.pop("furnishingstatus")
    input_dict["furnishingstatus_semi-furnished"] = 1 if furnishing =="semi-furnished"  else 0
    input_dict["furnishingstatus_unfurnished"] = 1 if furnishing == "unfurnished" else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]
    return {"predicted_price": round(float(prediction),3)}

