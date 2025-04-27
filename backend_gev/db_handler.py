import os
import sys
import datetime
from typing import Dict, Any

# Define DummyCollection for fallback when MongoDB isn't available
class DummyCollection:
    def __init__(self):
        pass
    def insert_one(self, doc):
        print(f"MOCK: Would insert document: {doc}")
        return type('obj', (object,), {'inserted_id': 'mock_id'})
    def find(self, *args, **kwargs):
        return []

# Add the parent directory to the path so we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the MongoDB connection
try:
    # Try both possible import paths
    try:
        # First try direct import 
        from backend.db_connection import db, collection, scan_stats_collection, DB_AVAILABLE
        print("MongoDB connection successfully imported from backend.db_connection")
    except ImportError:
        # Then try relative import - depends on where db_connection.py is located
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from db_connection import db, collection, scan_stats_collection, DB_AVAILABLE
        print("MongoDB connection successfully imported from db_connection")
except ImportError as e:
    print(f"Error importing MongoDB connection: {e}")
    DB_AVAILABLE = False
    # Create dummy objects for testing
    # Set up dummy db objects
    collection = DummyCollection()
    scan_stats_collection = DummyCollection()
    
# Create a new collection for statistics
if DB_AVAILABLE:
    print("Using MongoDB collection 'scan_statistics' for storing scan data")
    stats_collection = scan_stats_collection
else:
    print("WARNING: MongoDB not available - will use mock collection")
    stats_collection = DummyCollection()
    
def save_stats(stats: Dict[str, Any]) -> bool:
    """
    Save the current statistics to the MongoDB database
    
    Args:
        stats: Dictionary containing the scanning statistics
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    if not DB_AVAILABLE:
        print("Database connection not available")
        return False
    
    try:
        print(f"Attempting to save stats: {stats['total_emails']} emails, {stats['url_scans']} URLs")
        # Create a document with statistics and timestamp
        stats_doc = {
            "timestamp": datetime.datetime.now(),
            "email_stats": {
                "total_emails": stats.get("total_emails", 0),
                "phishing_emails": stats.get("phishing_emails", 0),
                "legitimate_emails": stats.get("legitimate_emails", 0),
                "detection_rate": stats.get("detection_rate", "0%")
            },
            "url_stats": {
                "total_urls": stats.get("url_scans", 0),
                "malicious_urls": stats.get("malicious_urls", 0)
            },
            "recent_detections": stats.get("recent_detections", [])
        }
        
        # Insert the document into the collection
        result = stats_collection.insert_one(stats_doc)
        print(f"Saved stats to database with ID: {result.inserted_id}")
        return True
        
    except Exception as e:
        print(f"Error saving stats to database: {e}")
        print(f"Stats that failed to save: {stats}")
        return False

def load_last_stats() -> Dict[str, Any]:
    """
    Load the most recent statistics from the database
    
    Returns:
        Dict: Dictionary containing the most recent statistics, or empty dict if not found
    """
    if not DB_AVAILABLE:
        print("Database connection not available - cannot load previous stats")
        return {}
    
    try:
        print("Attempting to load most recent stats from database...")
        # Find the most recent stats document
        cursor = stats_collection.find().sort("timestamp", -1).limit(1)
        
        # Convert cursor to list and get first item if available
        stats_list = list(cursor)
        if stats_list:
            latest_stats = stats_list[0]
            print(f"Loaded stats from {latest_stats.get('timestamp')}") 
            
            # Extract the email and URL statistics
            email_stats = latest_stats.get("email_stats", {})
            url_stats = latest_stats.get("url_stats", {})
            
            # Create a dictionary with the stats in the format expected by the API
            return {
                "total_emails": email_stats.get("total_emails", 0),
                "phishing_emails": email_stats.get("phishing_emails", 0),
                "legitimate_emails": email_stats.get("legitimate_emails", 0),
                "detection_rate": email_stats.get("detection_rate", "0%"),
                "url_scans": url_stats.get("total_urls", 0),
                "malicious_urls": url_stats.get("malicious_urls", 0),
                "recent_detections": latest_stats.get("recent_detections", [])
            }
        else:
            print("No previous stats found in database")
            return {}
            
    except Exception as e:
        print(f"Error loading stats from database: {e}")
        return {}
        
def get_historical_stats(days: int = 7) -> list:
    """
    Retrieve historical statistics from the database
    
    Args:
        days: Number of days to retrieve statistics for
        
    Returns:
        list: List of statistics documents
    """
    if not DB_AVAILABLE:
        print("Database connection not available")
        return []
    
    try:
        # Calculate the date for retrieving stats
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Query the database for statistics since start_date
        cursor = stats_collection.find(
            {"timestamp": {"$gte": start_date}},
            {"_id": 0}  # Exclude the _id field
        ).sort("timestamp", -1)  # Sort by timestamp descending
        
        # Convert cursor to list
        return list(cursor)
        
    except Exception as e:
        print(f"Error retrieving stats from database: {e}")
        return []

def load_recent_stats() -> Dict[str, Any]:
    """
    Load the most recent statistics from the database when the application starts
    
    Returns:
        Dict: Dictionary containing the most recent statistics, or empty dict if not found
    """
    return load_last_stats()