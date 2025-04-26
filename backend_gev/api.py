from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
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

# Neural Network model
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

def load_model(model_path):
    """Load the trained model from a .pth file"""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    input_size = checkpoint['input_size']
    # Check if we need to specify the number of classes
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    else:
        # Try to infer from the label_encoder
        num_classes = len(checkpoint.get('label_encoder', {}).classes_) if 'label_encoder' in checkpoint else 2
    
    model = PhishingDetectionModel(input_size, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    return model, checkpoint['vectorizer'], checkpoint['label_encoder']

# Load the model at startup
print("Loading the phishing detection model...")

# Possible model paths
possible_paths = [
    'phishing_model.pth',                  # Current directory
    'backend_gev/phishing_model.pth',      # From project root
    '../backend_gev/phishing_model.pth',   # One level up
]

MODEL_LOADED = False
for model_path in possible_paths:
    if os.path.exists(model_path):
        try:
            model, vectorizer, label_encoder = load_model(model_path)
            print(f"Model loaded successfully from {model_path}!")
            MODEL_LOADED = True
            break
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

if not MODEL_LOADED:
    print("Warning: Could not load the model from any known location.")
    print("The API will return errors until a valid model is available.")

def predict_email(email_text):
    """Predict if an email is phishing or legitimate"""
    # Preprocess the email text
    processed_text = preprocess_text(email_text)
    
    # Transform text using the same vectorizer
    features = vectorizer.transform([processed_text]).toarray()
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs, 1)
    
    # Convert numeric prediction to label
    predicted_label = label_encoder.inverse_transform(predicted.numpy())
    
    return predicted_label[0]

@app.route('/AI', methods=['POST'])
def analyze_email():
    """Endpoint to analyze if an email is ham or spam"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get the JSON data from the request
    data = request.get_json()
    
    # Check if email parameter is present
    if 'email' not in data:
        return jsonify({'error': 'Missing email parameter'}), 400
    
    email_text = data['email']
    
    # Make prediction
    try:
        result = predict_email(email_text)
        
        # Map result to ham/spam if needed (depending on your model's output labels)
        # Assuming the model's labels are already 'ham'/'spam' or 'legitimate'/'phishing'
        if result.lower() in ['legitimate', 'ham']:
            classification = 'ham'
        else:
            classification = 'spam'
            
        return jsonify({
            'result': classification,
            'original_label': result,
            'email': email_text[:100] + '...' if len(email_text) > 100 else email_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': MODEL_LOADED})

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 