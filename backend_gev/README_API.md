# Phishing Email Detection API

This directory contains a Flask API that serves a machine learning model for detecting phishing emails. The API provides a simple interface to check if an email is legitimate (ham) or phishing (spam).

## Setup and Installation

1. Install the required dependencies:

```bash
pip install flask flask-cors torch scikit-learn nltk pandas numpy requests
```

2. Make sure you have the trained model file `phishing_model.pth` in one of these locations:
   - Current directory (`backend_gev/phishing_model.pth`)
   - One level up from current directory (`../backend_gev/phishing_model.pth`)
   - Or in the root of your project (`phishing_model.pth`)

## Running the API Server

Start the Flask API server:

```bash
cd backend_gev
python api.py
```

The server will start on the default port 5000. You can access the API at:

- `http://localhost:5000/AI` - Main endpoint for checking emails
- `http://localhost:5000/health` - Health check endpoint

## Using the API

### 1. Using the command-line utility

The simplest way to check emails is using the `check_email.py` script:

```bash
# Check email from stdin (interactive)
python check_email.py --stdin

# Check email from a file
python check_email.py --file path/to/email.txt

# Check email from command line argument
python check_email.py --text "Your email content here"
```

### 2. Using the test script

The `test_api.py` script provides a comprehensive test with predefined examples and interactive testing:

```bash
python test_api.py
```

### 3. Using the API in your code

Use the `example_usage.py` script as a reference for integrating the API into your Python code:

```python
from example_usage import check_email_spam

# Check if an email is spam
is_spam, label, error = check_email_spam("Your email content here")

if error:
    print(f"Error: {error}")
else:
    print(f"Is spam: {is_spam}")
    print(f"Original label: {label}")
```

### 4. Calling the API directly

You can also call the API directly using HTTP requests:

```bash
curl -X POST http://localhost:5000/AI \
  -H "Content-Type: application/json" \
  -d '{"email": "Your email content here"}'
```

Python example using requests:

```python
import requests
import json

response = requests.post(
    "http://localhost:5000/AI",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"email": "Your email content here"})
)

result = response.json()
print(result)
```

## API Response Format

The API returns a JSON response with the following structure:

```json
{
  "result": "ham", // or "spam"
  "original_label": "legitimate", // original model prediction
  "email": "Preview of the email content..." // truncated for readability
}
```

## Error Handling

If an error occurs, the API returns a JSON response with an error message:

```json
{
  "error": "Error message here"
}
```

Common error scenarios:

- Model not loaded
- Missing email parameter
- Server error during prediction
