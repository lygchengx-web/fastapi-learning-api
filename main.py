import os

from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client


url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]

supabase = create_client(url, key)

app = FastAPI()


class InputData(BaseModel):
    x: float


def learning_function(x: float):
    return 2 * x


@app.post("/create")
def create_data(data: InputData):
    response = supabase.table("learning").insert({
        "x": data.x
    }).execute()

    return response.data

@app.get("/read")
def read_data():
    response = supabase.table("learning").select("*").execute()
    return response.data


@app.put("/update/{id}")
def update_data(id: int, data: InputData):
    response = supabase.table("learning").update({
        "x": data.x
    }).eq("id", id).execute()
    return response.data


@app.delete("/delete/{id}")
def delete_data(id: int):
    response = supabase.table("learning").delete().eq("id", id).execute()
    return response.data

@app.post("/predict")
def predict(data: InputData):
    result = learning_function(data.x)

    return {"result": result}
