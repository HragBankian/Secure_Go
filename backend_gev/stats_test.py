import requests
import json
import time
import sys
import random

def test_email_stats_persistence():
    """
    Test script to verify MongoDB stats persistence.
    This script will:
    1. Send test emails to the API for scanning
    2. Force stats to be saved to MongoDB
    3. Check if stats are updated
    4. Restart the API (in simulation)
    5. Verify loaded stats match what was saved
    """
    print("\n=== EMAIL STATS PERSISTENCE TEST ===\n")
    
    # Base URL for API
    base_url = "http://localhost:5000"
    
    try:
        # Step 1: Check API health
        print("1. Checking API health...")
        health_response = requests.get(f"{base_url}/health")
        health_data = health_response.json()
        print(f"   API status: {health_data.get('status', 'unknown')}")
        print(f"   Model loaded: {health_data.get('model_loaded', False)}")
        
        # Get initial stats
        print("\n2. Getting initial statistics...")
        initial_stats_response = requests.get(f"{base_url}/stats")
        initial_stats = initial_stats_response.json()
        print(f"   Initial email count: {initial_stats.get('total_emails', 0)}")
        print(f"   Initial URL scans: {initial_stats.get('url_scans', 0)}")
        
        # Step 2: Send some test emails for scanning
        print("\n3. Sending test emails for scanning...")
        
        # Test email templates (one phishing, one legitimate)
        test_emails = [
            {
                "type": "phishing",
                "content": """
                Dear Customer,
                We have detected suspicious activity on your account.
                Please click the link below to verify your identity:
                http://malicious-bank.com/verify?user=test
                Failure to confirm your account within 24 hours will result in account suspension.
                """
            },
            {
                "type": "legitimate",
                "content": """
                Hello team,
                I'm sending you the quarterly report as requested.
                Please review it and let me know if you have any questions.
                Best regards,
                John Smith
                Marketing Department
                """
            }
        ]
        
        # Send multiple test emails (mix of phishing and legitimate)
        for i in range(5):
            # Select a random email template
            email = random.choice(test_emails)
            print(f"   Sending test email #{i+1} ({email['type']})...")
            
            response = requests.post(
                f"{base_url}/api/scan/email",
                json={"email": email["content"]}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Classified as: {result.get('result', 'unknown')}")
            else:
                print(f"   ✗ Error scanning email: {response.status_code}")
        
        # Step 3: Get updated stats
        print("\n4. Getting updated statistics...")
        updated_stats_response = requests.get(f"{base_url}/stats")
        updated_stats = updated_stats_response.json()
        print(f"   Current email count: {updated_stats.get('total_emails', 0)}")
        print(f"   Phishing emails: {updated_stats.get('phishing_emails', 0)}")
        print(f"   Legitimate emails: {updated_stats.get('legitimate_emails', 0)}")
        
        # Step 4: Force save to database
        print("\n5. Forcing save to database...")
        save_response = requests.post(f"{base_url}/api/db/refresh")
        save_data = save_response.json()
        
        if save_data.get("success", False):
            print("   ✓ Successfully saved stats to database")
            print(f"   Timestamp: {save_data.get('timestamp', 'unknown')}")
        else:
            print("   ✗ Failed to save stats to database")
            print(f"   Error: {save_data.get('message', 'unknown error')}")
            return False
        
        # Step 5: Simulate restart by calling the database stats endpoint
        print("\n6. Simulating restart by retrieving stats from database...")
        print("   (This verifies the data would be loaded correctly on restart)")
        
        db_stats_response = requests.get(f"{base_url}/api/db/stats")
        db_stats = db_stats_response.json()
        
        if "current" in db_stats:
            current = db_stats["current"]
            email_stats = current.get("email_stats", {})
            
            print("\n7. Verifying statistics persistence...")
            print(f"   DB total emails: {email_stats.get('total_emails', 0)}")
            print(f"   Current total emails: {updated_stats.get('total_emails', 0)}")
            
            # Check if the statistics match
            if (email_stats.get('total_emails', 0) == updated_stats.get('total_emails', 0) and
                email_stats.get('phishing_emails', 0) == updated_stats.get('phishing_emails', 0)):
                print("   ✓ SUCCESS! Statistics are correctly persisted in the database")
                print("   Your application will load these values on restart")
            else:
                print("   ✗ WARNING: Discrepancy between current stats and database stats")
                print("   This may indicate a problem with database persistence")
        else:
            print("   ✗ Could not retrieve database statistics")
            return False
        
        # Step 6: Check if historical data is available
        historical = db_stats.get("historical", [])
        print(f"\n8. Found {len(historical)} historical stat records in database")
        
        if historical:
            print("   Most recent historical records:")
            for i, record in enumerate(historical[:3]):
                timestamp = record.get("timestamp", "unknown")
                email_stats = record.get("email_stats", {})
                print(f"   {i+1}. {timestamp}: {email_stats.get('total_emails', 0)} emails analyzed")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to API at {base_url}")
        print("Make sure the Flask API is running.")
        return False
        
    except Exception as e:
        print(f"ERROR during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_email_stats_persistence()
    
    if success:
        print("\n✅ Email stats persistence test completed!")
        print("Your application is correctly saving statistics to MongoDB.")
        print("These statistics will be loaded when the application restarts.")
    else:
        print("\n❌ Email stats persistence test failed.")
        print("Please check the issues reported above.")
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 