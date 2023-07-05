from fastapi import FastAPI
from enum import Enum

app = FastAPI()

class AvailableFoods (str, Enum):
    indian = 'indian',
    american = 'american',
    bengali = 'bengali'

food_items={
    'indian': ['Somosha', 'Dosa'],
    'american': ['hot dog','chocolate'],
    'bengali': ['Rice', 'Vegatble']
}

@app.get("/hello/{food_name}")

async def roy(food_name: AvailableFoods):
    return food_items.get(food_name)

coupon_code={
    1: '10%',
    2: '20%',
    3: '30%'
}

@app.get("/get_coupon/{code}")
async def get_item(code: int):
    return {"discount_amount": coupon_code.get(code)}
