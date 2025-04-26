import torch
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

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
        
        # Simple tokenization using split() instead of nltk.word_tokenize
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
    checkpoint = torch.load(model_path)
    input_size = checkpoint['input_size']
    model = PhishingDetectionModel(input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    return model, checkpoint['vectorizer'], checkpoint['label_encoder']

def predict_email(email_text, model, vectorizer, label_encoder):
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

def main():
    # Load the trained model
    try:
        model, vectorizer, label_encoder = load_model('backend_gev/phishing_model.pth')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Example emails (simulated)
    legitimate_email = """
    Hello John,
    Just wanted to follow up on our meeting last week. I've attached the presentation slides as promised.
    Let's schedule a follow-up call next Tuesday at 2pm.
    Best regards,
    Alice Johnson
    Marketing Director
    """
    
    phishing_email = """
    URGENT: Your account has been compromised!
    Dear Customer,
    We have detected suspicious activity on your account. Click the link below to verify your identity and restore access immediately:
    http://secure-account-verify.com/login
    If you don't verify within 24 hours, your account will be suspended.
    """
    
    # Predict
    print("\n===== Legitimate Email Example =====")
    print(legitimate_email)
    prediction = predict_email(legitimate_email, model, vectorizer, label_encoder)
    print(f"\nPrediction: {prediction}")
    
    print("\n===== Phishing Email Example =====")
    print(phishing_email)
    prediction = predict_email(phishing_email, model, vectorizer, label_encoder)
    print(f"\nPrediction: {prediction}")
    
    # Interactive testing
    print("\n===== Try your own emails =====")
    while True:
        test_email = input("\nEnter an email to analyze (or 'q' to quit): ")
        if test_email.lower() == 'q':
            break
        prediction = predict_email(test_email, model, vectorizer, label_encoder)
        print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main() 