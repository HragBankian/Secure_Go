# Phishing Email Detection Model

This directory contains a machine learning model for detecting phishing emails using PyTorch.

## Requirements

- Python 3.6+
- PyTorch
- scikit-learn
- NLTK
- pandas
- numpy

You can install all required packages using:

```bash
pip install torch pandas numpy scikit-learn nltk
```

## Files

- `mail_data_1.csv`, `mail_data_2.csv` - Email datasets for training the model
- `phishing_model.py` - Script to train the model and save it as .pth file
- `predict_phishing.py` - Script to load the trained model and make predictions
- `phishing_model.pth` - Trained model file (will be created after running the training script)

## Usage

### Step 1: Train the Model

Run the following command to train the model:

```bash
python phishing_model.py
```

This will:

- Load and preprocess the email datasets
- Extract features using TF-IDF
- Train a neural network model
- Save the trained model as `phishing_model.pth`

### Step 2: Use the Model for Prediction

After training, you can use the model to predict if new emails are phishing attempts:

```bash
python predict_phishing.py
```

This script:

- Loads the trained model
- Includes examples of legitimate and phishing emails
- Provides an interactive prompt to test your own emails

## Model Architecture

The model uses:

- Text preprocessing with NLTK (lowercasing, removing HTML tags, URLs, special characters)
- TF-IDF vectorization for feature extraction
- A neural network with:
  - Input layer (TF-IDF features)
  - Hidden layer with ReLU activation
  - Dropout for regularization
  - Second hidden layer
  - Output layer (2 classes: phishing or legitimate)

## Integration

To integrate this model into your application:

1. Make sure you have the trained `phishing_model.pth` file
2. Use the `load_model()` and `predict_email()` functions from `predict_phishing.py`
3. Pass email text to the `predict_email()` function to get the prediction

Example:

```python
from predict_phishing import load_model, predict_email

# Load the model
model, vectorizer, label_encoder = load_model('path/to/phishing_model.pth')

# Make predictions
email_text = "Your email text here..."
prediction = predict_email(email_text, model, vectorizer, label_encoder)
print(f"This email is: {prediction}")
```
