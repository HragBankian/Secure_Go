from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["SecureGoDB"]
collection = db["user_data"]

app = FastAPI()

class UserData(BaseModel):
    name: str
    email: str
    role: str

@app.post("/users/")
def create_user(user: UserData):
    result = collection.insert_one(user.dict())
    return {"id": str(result.inserted_id)}

@app.get("/users/")
def get_users():
    users = list(collection.find())
    for u in users:
        u["_id"] = str(u["_id"])
    return users

@app.put("/users/{user_id}")
def update_user(user_id: str, user: UserData):
    result = collection.update_one({"_id": ObjectId(user_id)}, {"$set": user.dict()})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"updated": result.modified_count}

@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    result = collection.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": result.deleted_count}
