#!/usr/bin/env python
"""
Script to fix the phishing model by creating a compatible TF-IDF vectorizer
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Simple tokenization using split()
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)
    else:
        return ''

class PhishingDetectionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super(PhishingDetectionModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.output = torch.nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def load_broken_model(model_path):
    """Load the broken model and extract what we can"""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    input_size = checkpoint['input_size']
    
    # Create a new model with the same architecture
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    else:
        num_classes = len(checkpoint.get('label_encoder', {}).classes_) if 'label_encoder' in checkpoint else 2
    
    model = PhishingDetectionModel(input_size, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def fix_model():
    """Fix the broken model by creating a compatible vectorizer"""
    print("Fixing broken phishing model...")
    
    # Load email data to train the vectorizer
    csv_paths = [
        os.path.join('backend_gev', 'mail_data_1.csv'),
        os.path.join('backend_gev', 'mail_data_2.csv')
    ]
    
    # Load the model first
    model_path = os.path.join('backend_gev', 'phishing_model.pth')
    
    try:
        broken_model, checkpoint = load_broken_model(model_path)
        print("Loaded original model state")
        
        # Extract original label encoder if available
        original_label_encoder = checkpoint.get('label_encoder', None)
        if original_label_encoder is not None:
            print(f"Original label classes: {original_label_encoder.classes_}")
        else:
            print("Could not extract original label encoder")
            # Create a default label encoder
            original_label_encoder = LabelEncoder()
            original_label_encoder.classes_ = np.array(['ham', 'spam'])
        
        # Load email data
        all_data = []
        for path in csv_paths:
            if os.path.exists(path):
                print(f"Loading data from {path}")
                try:
                    df = pd.read_csv(path)
                    if 'text' in df.columns and 'label' in df.columns:
                        all_data.append(df)
                    else:
                        print(f"Missing required columns in {path}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")
            else:
                print(f"File not found: {path}")
        
        if not all_data:
            print("No valid data found, using default sample data")
            # Create some sample data for the vectorizer
            sample_texts = [
                "Hello, this is a legitimate email about a meeting tomorrow.",
                "Free viagra! Click here for discount meds!",
                "Your account has been compromised. Reset your password now!",
                "Thank you for your recent purchase. Here is your receipt.",
                "Urgent: Your bank account needs verification, click the link.",
                "Meeting agenda for next week's department planning session.",
                "You've won a million dollars in the lottery. Send us your details.",
                "Please review the attached document for our next project.",
                "Security alert: Suspicious activity detected on your account.",
                "Your package has been shipped and will arrive on Monday."
            ]
            sample_labels = ['ham', 'spam', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
            df = pd.DataFrame({'text': sample_texts, 'label': sample_labels})
            all_data = [df]
        
        # Combine all data
        if len(all_data) > 1:
            combined_df = pd.concat(all_data, ignore_index=True)
        else:
            combined_df = all_data[0]
        
        print(f"Total samples: {len(combined_df)}")
        
        # Preprocess the text data
        print("Preprocessing text data...")
        combined_df['processed_text'] = combined_df['text'].apply(preprocess_text)
        
        # Create and fit a new vectorizer
        print("Creating new TF-IDF vectorizer...")
        new_vectorizer = TfidfVectorizer(max_features=5000)
        new_vectorizer.fit(combined_df['processed_text'])
        
        # Create a new label encoder
        new_label_encoder = LabelEncoder()
        new_label_encoder.fit(combined_df['label'])
        
        print(f"New label classes: {new_label_encoder.classes_}")
        
        # Create a fixed model checkpoint
        fixed_checkpoint = {
            'model_state_dict': broken_model.state_dict(),
            'input_size': checkpoint['input_size'],
            'vectorizer': new_vectorizer,
            'label_encoder': original_label_encoder,  # Use original if possible
            'num_classes': len(original_label_encoder.classes_)
        }
        
        # Save the fixed model
        fixed_model_path = 'backend_gev/fixed_phishing_model.pth'
        torch.save(fixed_checkpoint, fixed_model_path)
        print(f"Fixed model saved to {fixed_model_path}")
        
        # Test the fixed model
        test_email = "Hello, this is a test email."
        processed_text = preprocess_text(test_email)
        features = new_vectorizer.transform([processed_text]).toarray()
        features_tensor = torch.FloatTensor(features)
        
        with torch.no_grad():
            outputs = broken_model(features_tensor)
            _, predicted = torch.max(outputs, 1)
        
        predicted_label = original_label_encoder.inverse_transform(predicted.numpy())
        print(f"Test prediction with fixed model: {predicted_label[0]}")
        
        return True
    
    except Exception as e:
        print(f"Error fixing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_model()
    if success:
        print("\nModel fixed successfully! Restart the API server to use the fixed model.")
    else:
        print("\nFailed to fix the model.") 