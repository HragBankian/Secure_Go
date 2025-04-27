import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI from environment variables or use a default
# Replace '<db_password>' placeholder with a literal string to avoid confusion
uri = os.getenv("MONGO_URI")
if uri and '<db_password>' in uri:
    print("WARNING: Your MongoDB connection string still contains '<db_password>' placeholder.")
    print("Please replace it with your actual password in the .env file.")
    # Use a default local connection if the password isn't set correctly
    uri = "mongodb://localhost:27017/"

# Print connection information (without revealing full password)
if uri:
    # Create a safe version of URI for logging (hide actual password)
    masked_uri = uri
    if '@' in uri and ':' in uri:
        prefix = uri.split('@')[0]
        if ':' in prefix:
            username_part = prefix.split(':')[0]
            masked_uri = f"{username_part}:****@{uri.split('@')[1]}"
    print(f"Connecting to MongoDB using: {masked_uri}")
else:
    uri = "mongodb://localhost:27017/"
    print(f"No MONGO_URI found, connecting to: {uri}")

try:
    # Connect to MongoDB with a timeout to avoid hanging
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    
    # Test connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
    
    # Set up database and collections
    db = client["SecureGoDB"]
    collection = db["user_data"]
    scan_stats_collection = db["scan_statistics"]
    
    # Check if we're connected to Atlas or a local instance
    is_atlas = "mongodb+srv" in uri
    print(f"Connected to {'MongoDB Atlas' if is_atlas else 'local MongoDB'}")
    
    # Create a test document if the user_data collection is empty
    if collection.count_documents({}) == 0:
        print("Creating a test document in user_data collection")
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "role": "user",
            "created_at": "2023-11-15T12:00:00.000Z"
        }
        result = collection.insert_one(user_data)
        print(f"Test document created with ID: {result.inserted_id}")
    else:
        print(f"Found {collection.count_documents({})} documents in user_data collection")
        
    DB_AVAILABLE = True
    
except Exception as e:
    print(f"MongoDB connection error: {e}")
    print("Creating fallback database objects - data will NOT be persisted")
    
    # Define fallback classes for graceful degradation
    class DummyCollection:
        def __init__(self, name):
            self.name = name
            self.data = []
            
        def insert_one(self, document):
            self.data.append(document)
            print(f"[MOCK] Inserted document into {self.name}")
            return type('obj', (object,), {'inserted_id': 'mock_id_' + str(len(self.data))})
            
        def find(self, query=None, projection=None):
            print(f"[MOCK] Querying {self.name} with {query}")
            return []
            
        def count_documents(self, query=None):
            return 0
            
        def find_one(self, query=None):
            return None
            
        def update_one(self, query, update, upsert=False):
            print(f"[MOCK] Updating document in {self.name}")
            return type('obj', (object,), {'modified_count': 0})
    
    class DummyDB:
        def __init__(self, name):
            self.name = name
            self.collections = {}
            
        def __getitem__(self, key):
            if key not in self.collections:
                self.collections[key] = DummyCollection(key)
            return self.collections[key]
            
        def list_collection_names(self):
            return list(self.collections.keys())
    
    # Create dummy objects
    db = DummyDB("SecureGoDB")
    collection = db["user_data"]
    scan_stats_collection = db["scan_statistics"]
    DB_AVAILABLE = False
