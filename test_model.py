#!/usr/bin/env python
"""
Simple test script to verify the model loading and prediction functionality
"""

import os
import sys
import torch

# Add the backend_gev directory to the path
sys.path.append("backend_gev")

# Try to import the model functions
try:
    from backend_gev.predict_phishing import load_model, predict_email
    print("Successfully imported prediction functions")
except ImportError as e:
    print(f"Error importing prediction functions: {e}")
    # Try an alternative import
    try:
        import predict_phishing
        print("Successfully imported predict_phishing module")
    except ImportError as e:
        print(f"Error importing predict_phishing module: {e}")
        sys.exit(1)

def test_model_loading():
    """Test if the model can be loaded correctly"""
    model_paths = [
        'backend_gev/phishing_model.pth',
        'phishing_model.pth',
    ]
    
    for path in model_paths:
        try:
            print(f"Trying to load model from: {path}")
            if not os.path.exists(path):
                print(f"  - File does not exist: {path}")
                continue
                
            # Try to load the model
            model, vectorizer, label_encoder = load_model(path)
            print(f"  - Model loaded successfully from {path}")
            print(f"  - Model type: {type(model)}")
            print(f"  - Vectorizer type: {type(vectorizer)}")
            print(f"  - Label encoder type: {type(label_encoder)}")
            
            # Try a test prediction
            test_email = "Hello, this is a test email to check for phishing detection."
            prediction = predict_email(test_email, model, vectorizer, label_encoder)
            print(f"  - Test prediction result: {prediction}")
            
            return True, model, vectorizer, label_encoder
        except Exception as e:
            print(f"  - Error loading model from {path}: {e}")
    
    return False, None, None, None

def test_api_prediction(model=None, vectorizer=None, label_encoder=None):
    """Test if we can replicate the API's prediction behavior"""
    import requests
    import json
    
    # Test the API directly
    try:
        print("\nTesting API directly...")
        response = requests.post(
            "http://localhost:5000/AI",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"email": "Hello, this is a test email to check for phishing detection."})
        )
        print(f"API response status: {response.status_code}")
        print(f"API response: {response.text}")
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    print("Starting model test...")
    success, model, vectorizer, label_encoder = test_model_loading()
    
    if success:
        print("\nModel loading successful!")
        test_api_prediction(model, vectorizer, label_encoder)
    else:
        print("\nModel loading failed!")
        test_api_prediction()  # Still try the API test 