#!/usr/bin/env python
"""
Example of how to use the phishing email detection API from Python code.
"""

import requests
import json

def check_email_spam(email_text, api_url="http://localhost:5000/AI"):
    """
    Check if an email is ham or spam using the API.
    
    Args:
        email_text (str): The text of the email to check
        api_url (str): The API endpoint URL
        
    Returns:
        tuple: (is_spam, label, error_message)
            - is_spam (bool): True if email is classified as spam, False if ham, None on error
            - label (str): Original label from the model
            - error_message (str): Error message if any, None if successful
    """
    try:
        # Make API request
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"email": email_text})
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            is_spam = result['result'].lower() == 'spam'
            label = result.get('original_label', 'unknown')
            return is_spam, label, None
        else:
            return None, None, f"API error: {response.status_code} - {response.text}"
    
    except requests.exceptions.ConnectionError:
        return None, None, "Cannot connect to API. Check if the server is running."
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def main():
    # Example emails
    emails = [
        {
            "name": "Legitimate Email",
            "text": """
            Hello John,
            Just wanted to follow up on our meeting last week. I've attached the presentation slides as promised.
            Let's schedule a follow-up call next Tuesday at 2pm.
            Best regards,
            Alice Johnson
            """
        },
        {
            "name": "Phishing Email",
            "text": """
            URGENT: Your account has been compromised!
            Dear Customer,
            We have detected suspicious activity on your account. Click the link below to verify your identity:
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
        return
    
    # Test with examples
    for email in emails:
        print(f"Email: {email['name']}")
        print("-" * 50)
        
        is_spam, label, error = check_email_spam(email['text'])
        
        if error:
            print(f"Error: {error}")
        else:
            print(f"Classification: {'Spam' if is_spam else 'Ham (Legitimate)'}")
            print(f"Original Label: {label}")
        
        print("\n" + "=" * 50 + "\n")
    
    # Custom input example
    print("Now you can test your own email:")
    custom_email = input("Enter email text: ")
    is_spam, label, error = check_email_spam(custom_email)
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"\nClassification: {'Spam' if is_spam else 'Ham (Legitimate)'}")
        print(f"Original Label: {label}")

if __name__ == "__main__":
    main() 