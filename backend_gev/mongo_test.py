import os
import sys
import datetime
import time
import uuid
from pymongo import MongoClient
from dotenv import load_dotenv

def test_mongodb_connection():
    """
    Test script to verify MongoDB connection and data persistence.
    This script will:
    1. Connect to MongoDB using credentials from .env
    2. Insert a test document with a unique ID
    3. Query to verify the document was saved
    4. Report success or failure
    """
    print("\n=== MongoDB Connection Test ===\n")
    
    # Step 1: Load connection string from environment
    load_dotenv()  # Try to load from .env file
    uri = os.getenv("MONGO_URI")
    
    if not uri:
        print("❌ ERROR: No MONGO_URI found in .env file")
        print("Please ensure your .env file contains the MONGO_URI variable")
        return False
    
    # Create a safe version of URI for logging (hide password)
    masked_uri = uri
    if '@' in uri and ':' in uri:
        prefix = uri.split('@')[0]
        if ':' in prefix:
            username_part = prefix.split(':')[0]
            masked_uri = f"{username_part}:****@{uri.split('@')[1]}"
    
    print(f"🔌 Connecting to MongoDB: {masked_uri}")
    
    try:
        # Step 2: Connect to MongoDB
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test connection with a ping
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
        
        # Get database and collection
        db = client["SecureGoDB"]
        test_collection = db["connection_tests"]
        
        # Step 3: Create a unique test document
        test_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()
        test_doc = {
            "test_id": test_id,
            "timestamp": timestamp,
            "message": "This is a test document",
            "app_version": "1.0.0",
            "test_type": "connection_verification"
        }
        
        print(f"\n📝 Inserting test document with ID: {test_id}")
        result = test_collection.insert_one(test_doc)
        print(f"✅ Document inserted with MongoDB ID: {result.inserted_id}")
        
        # Step 4: Verify the document was saved by querying for it
        print("\n🔍 Verifying document was saved...")
        # Short pause to ensure data is properly saved
        time.sleep(1)
        
        found_doc = test_collection.find_one({"test_id": test_id})
        
        if found_doc:
            print(f"✅ SUCCESS! Document was successfully saved and retrieved")
            print(f"   - Retrieved timestamp: {found_doc.get('timestamp')}")
            print(f"   - MongoDB document ID: {found_doc.get('_id')}")
            
            # Step 5: List some recent tests
            print("\n📊 Recent connection tests:")
            recent_tests = list(test_collection.find().sort("timestamp", -1).limit(5))
            for i, test in enumerate(recent_tests):
                print(f"   {i+1}. {test.get('timestamp')} - {test.get('test_id')}")
            
            print(f"\n📊 Total connection test records: {test_collection.count_documents({})}")
            return True
        else:
            print("❌ ERROR: Document not found after insertion!")
            return False
    
    except Exception as e:
        print(f"❌ MongoDB Error: {e}")
        print("\nPossible causes:")
        print("1. Your MongoDB connection string might be incorrect")
        print("2. Your MongoDB username or password might be incorrect")
        print("3. The MongoDB server might be unavailable")
        print("4. Network connectivity issues")
        print("\nPlease check your .env file and verify your MongoDB credentials.")
        return False

if __name__ == "__main__":
    success = test_mongodb_connection()
    
    if success:
        print("\n🎉 MongoDB connection test completed successfully!")
        print("Your application should be able to save and retrieve data from MongoDB.")
    else:
        print("\n❌ MongoDB connection test failed.")
        print("Please fix the issues above before proceeding.")
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 