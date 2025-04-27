import requests
import json
import time
import sys

def test_immediate_save():
    """
    This script tests if stats are immediately saved to MongoDB after each request.
    It sends a single email for scanning, then checks if the stats were immediately saved
    by verifying them through the stats endpoint.
    """
    print("\n=== TESTING IMMEDIATE STATS SAVING ===\n")
    
    # Base URL for API
    base_url = "http://localhost:5000"
    
    try:
        # Step 1: Get initial stats
        print("1. Getting initial statistics...")
        initial_stats_response = requests.get(f"{base_url}/stats")
        initial_stats = initial_stats_response.json()
        initial_email_count = initial_stats.get('total_emails', 0)
        print(f"   Initial email count: {initial_email_count}")
        
        # Step 2: Send a test email for scanning
        print("\n2. Sending a test email for scanning...")
        
        test_email = """
        Dear Customer,
        We have detected suspicious activity on your account.
        Please click the link below to verify your identity:
        http://malicious-bank.com/verify?user=test
        Failure to confirm your account within 24 hours will result in account suspension.
        """
        
        response = requests.post(
            f"{base_url}/api/scan/email",
            json={"email": test_email}
        )
        
        if response.status_code != 200:
            print(f"   ❌ Error scanning email: {response.status_code}")
            return False
            
        result = response.json()
        print(f"   ✅ Email classified as: {result.get('result', 'unknown')}")
        
        # Step 3: Check if stats were updated
        print("\n3. Checking if stats were updated...")
        updated_stats_response = requests.get(f"{base_url}/stats")
        updated_stats = updated_stats_response.json()
        new_email_count = updated_stats.get('total_emails', 0)
        print(f"   New email count: {new_email_count}")
        
        if new_email_count > initial_email_count:
            print(f"   ✅ Stats were updated (Email count increased by {new_email_count - initial_email_count})")
        else:
            print("   ❌ Stats were not updated")
            return False
        
        # Step 4: Fetch data from MongoDB through the db stats endpoint
        print("\n4. Fetching stats from MongoDB to verify immediate saving...")
        db_stats_response = requests.get(f"{base_url}/api/db/stats")
        db_stats = db_stats_response.json()
        
        if "current" not in db_stats:
            print("   ❌ Could not retrieve database statistics")
            return False
            
        current = db_stats.get("current", {})
        email_stats = current.get("email_stats", {})
        db_email_count = email_stats.get("total_emails", 0)
        
        print(f"   Email count in MongoDB: {db_email_count}")
        
        # Step 5: Verify that MongoDB stats match the current stats
        print("\n5. Verifying that MongoDB stats match current stats...")
        
        if db_email_count == new_email_count:
            print("   ✅ SUCCESS! Stats were immediately saved to MongoDB")
            # Now check historical records
            historical = db_stats.get("historical", [])
            print(f"\n   MongoDB contains {len(historical)} historical stats records")
            return True
        else:
            print("   ❌ Stats were not immediately saved to MongoDB")
            print(f"   Current API stats: {new_email_count} emails")
            print(f"   MongoDB stats: {db_email_count} emails")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Could not connect to API at {base_url}")
        print("Make sure the Flask API is running.")
        return False
        
    except Exception as e:
        print(f"ERROR during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_immediate_save()
    
    if success:
        print("\n✅ Immediate save test completed successfully!")
        print("Your stats are being saved to MongoDB immediately after each request.")
    else:
        print("\n❌ Immediate save test failed.")
        print("Stats are not being saved to MongoDB immediately after each request.")
    
    sys.exit(0 if success else 1) 