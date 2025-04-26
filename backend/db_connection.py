import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGO_URI")
client = MongoClient(uri)

db = client["SecureGoDB"]
collection = db["user_data"]

user_data = {
    "name": "Dem",
    "email": "greg@example.com",
    "role": "developer"
}

# Insert the data
result = collection.insert_one(user_data)

print(f"Inserted document ID: {result.inserted_id}")
