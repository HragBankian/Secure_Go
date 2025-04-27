import requests
import json
import time
import sys

def test_db_connection():
    print("Testing database connection and stats saving functionality...")
    
    # Base URL for API
    base_url = "http://localhost:5000"
    
    try:
        # Test the health endpoint first
        print("\n1. Checking API health...")
        health_response = requests.get(f"{base_url}/health")
        health_data = health_response.json()
        print(f"Health status: {health_data.get('status', 'unknown')}")
        print(f"Model loaded: {health_data.get('model_loaded', False)}")
        
        # Test the manual stats save endpoint
        print("\n2. Trying to save stats to database...")
        save_response = requests.post(f"{base_url}/api/db/save-stats")
        save_data = save_response.json()
        print(f"Save response: {json.dumps(save_data, indent=2)}")
        
        if save_response.status_code == 503:
            print("\nWARNING: Database functionality is not available!")
            print("Check if MongoDB is running and connection is properly configured.")
            print("Make sure the MongoDB URI is set in your .env file.")
            sys.exit(1)
        
        # Test the stats retrieval endpoint
        print("\n3. Retrieving stats from database...")
        stats_response = requests.get(f"{base_url}/api/db/stats")
        stats_data = stats_response.json()
        
        if "error" in stats_data:
            print(f"Error retrieving stats: {stats_data['error']}")
        else:
            print("Current stats summary:")
            current = stats_data.get("current", {})
            email_stats = current.get("email_stats", {})
            url_stats = current.get("url_stats", {})
            
            print(f"- Total emails analyzed: {email_stats.get('total_emails', 0)}")
            print(f"- Phishing emails detected: {email_stats.get('phishing_emails', 0)}")
            print(f"- Total URLs scanned: {url_stats.get('total_urls', 0)}")
            print(f"- Malicious URLs detected: {url_stats.get('malicious_urls', 0)}")
            
            # Check if we have historical data
            historical = stats_data.get("historical", [])
            print(f"\nHistorical records in database: {len(historical)}")
            if historical:
                print("Recent historical records:")
                for i, record in enumerate(historical[:3]):  # Show the 3 most recent records
                    timestamp = record.get("timestamp", "unknown")
                    email_stats = record.get("email_stats", {})
                    print(f"{i+1}. {timestamp}: {email_stats.get('total_emails', 0)} emails analyzed")
        
        # Test the enhanced metrics endpoint
        print("\n4. Checking enhanced metrics (includes database status)...")
        metrics_response = requests.get(f"{base_url}/api/metrics")
        metrics_data = metrics_response.json()
        db_status = metrics_data.get("database", {})
        print(f"Database available: {db_status.get('available', False)}")
        print(f"Last database save: {db_status.get('last_save_attempt', 'Never')}")
        
        print("\n✅ Database integration test completed!")
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to API at {base_url}")
        print("Make sure the Flask API is running.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_db_connection() 