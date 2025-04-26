import requests
import json

def test_phishing_detection_api():
    """
    Test the phishing detection API by sending example emails
    """
    # API endpoint
    url = "http://localhost:5000/AI"
    
    # Test cases
    test_cases = [
        {
            "name": "Legitimate Email",
            "email": """
            Hello John,
            Just wanted to follow up on our meeting last week. I've attached the presentation slides as promised.
            Let's schedule a follow-up call next Tuesday at 2pm.
            Best regards,
            Alice Johnson
            Marketing Director
            """
        },
        {
            "name": "Phishing Email",
            "email": """
            URGENT: Your account has been compromised!
            Dear Customer,
            We have detected suspicious activity on your account. Click the link below to verify your identity and restore access immediately:
            http://secure-account-verify.com/login
            If you don't verify within 24 hours, your account will be suspended.
            """
        }
    ]
    
    print("Testing Phishing Detection API\n")
    
    # Check if API is running
    try:
        health_response = requests.get("http://localhost:5000/health")
        health_data = health_response.json()
        print(f"API Status: {health_data['status']}")
        print(f"Model Loaded: {health_data['model_loaded']}\n")
        
        if not health_data['model_loaded']:
            print("Warning: Model is not loaded in the API. Results may be incorrect.")
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to the API. Make sure the Flask server is running.\n")
        print("Run 'python api.py' in a separate terminal to start the server.")
        return
    
    # Run test cases
    for i, test_case in enumerate(test_cases):
        print(f"Test Case {i+1}: {test_case['name']}")
        print("-" * 50)
        print(f"Email: {test_case['email'][:100]}...")
        
        # Make API request
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"email": test_case['email']})
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                print(f"\nResult: {result['result']}")
                print(f"Original Label: {result['original_label']}")
            else:
                print(f"\nError: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"\nException: {str(e)}")
        
        print("\n" + "=" * 50 + "\n")
    
    # Interactive testing
    print("Interactive Testing")
    print("-" * 50)
    print("Enter an email to analyze (or 'q' to quit):")
    
    while True:
        email_text = input("\nEmail: ")
        if email_text.lower() == 'q':
            break
        
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"email": email_text})
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Result: {result['result']}")
                print(f"Original Label: {result['original_label']}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Exception: {str(e)}")

if __name__ == "__main__":
    test_phishing_detection_api() 