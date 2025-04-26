#!/usr/bin/env python
"""
Simple script to check if an email is ham or spam using the API.
Usage:
    python check_email.py "Your email text here"
    python check_email.py --file email.txt
"""

import requests
import json
import sys
import argparse

def check_email(email_text, api_url="http://localhost:5000/AI"):
    """
    Send email text to the API and get the ham/spam prediction.
    
    Args:
        email_text (str): The email content to check
        api_url (str): The API endpoint URL
    
    Returns:
        dict: The API response
    """
    try:
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"email": email_text})
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status code {response.status_code}", "details": response.text}
    
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API. Make sure the Flask server is running."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check if an email is ham or spam")
    
    # Create a mutually exclusive group for input methods
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Email text to check")
    group.add_argument("--file", type=str, help="File containing email text")
    group.add_argument("--stdin", action="store_true", help="Read email text from standard input")
    
    parser.add_argument("--url", type=str, default="http://localhost:5000/AI", 
                        help="API endpoint URL (default: http://localhost:5000/AI)")
    
    args = parser.parse_args()
    
    # Get email text based on the input method
    if args.text:
        email_text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as file:
                email_text = file.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return 1
    elif args.stdin:
        print("Enter email text (Ctrl+D or Ctrl+Z when finished):")
        email_text = sys.stdin.read()
    
    # Check if API is running
    try:
        health_response = requests.get(args.url.replace('/AI', '/health'))
        health_data = health_response.json()
        
        if not health_data.get('model_loaded', False):
            print("Warning: Model is not loaded in the API. Results may be incorrect.")
    except:
        pass

    # Send the email to the API
    result = check_email(email_text, args.url)
    
    # Print the result
    if "error" in result:
        print(f"Error: {result['error']}")
        if "details" in result:
            print(f"Details: {result['details']}")
    else:
        print(f"\nEmail Classification: {result['result']}")
        print(f"Original Label: {result.get('original_label', 'N/A')}")
        
        # Print a sample of the input email for reference
        email_preview = email_text[:100] + ('...' if len(email_text) > 100 else '')
        print(f"\nEmail Preview: {email_preview}")
    
    return 0

if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        sys.argv.append("--stdin")
    
    sys.exit(main()) 