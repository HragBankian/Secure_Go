from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import datetime
import pickle
import threading
import functools
import time
from collections import OrderedDict
from typing import Dict, List, Tuple, Any

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

# Track some basic stats
stats = {
    'total_emails': 0,
    'phishing_emails': 0,
    'legitimate_emails': 0,
    'detection_rate': '0%',
    'recent_detections': [],
    'url_scans': 0,
    'malicious_urls': 0,
    'start_time': datetime.datetime.now().isoformat(),
    'api_version': '1.1.0'  # Track API version
}

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Cache for URL scanning results - OrderedDict works as LRU cache
url_cache: OrderedDict = OrderedDict()
URL_CACHE_SIZE = 10000  # Maximum number of URLs to cache
URL_CACHE_LOCK = threading.Lock()  # Thread safety for cache access

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

# Create a simple LRU cache decorator
def lru_cache(maxsize=128):
    cache = OrderedDict()
    lock = threading.RLock()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            with lock:
                if key in cache:
                    # Move to end to signify recent use
                    cache.move_to_end(key)
                    return cache[key]
                result = func(*args, **kwargs)
                cache[key] = result
                if len(cache) > maxsize:
                    cache.popitem(last=False)  # Remove oldest item
                return result
        return wrapper
    return decorator

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
    'backend_gev/fixed_phishing_model.pth',  # Our fixed model (try first)
    'fixed_phishing_model.pth',              # Fixed model in current directory
    'phishing_model.pth',                    # Original model in current directory
    'backend_gev/phishing_model.pth',        # Original model from project root
    '../backend_gev/phishing_model.pth',     # Original model one level up
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

# Custom template filter for current year
@app.template_filter('now')
def filter_now(format_string):
    if format_string == 'year':
        return datetime.datetime.now().year
    return datetime.datetime.now().strftime(format_string)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', stats=stats)

# API endpoint to analyze emails
@app.route('/AI', methods=['POST'])
@app.route('/api/scan/email', methods=['POST'])  # New standardized endpoint
def analyze_email():
    """Endpoint to analyze if an email is ham or spam"""
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get the JSON data from the request
    data = request.get_json()
    
    # Check if email parameter is present (handle both formats)
    email_text = None
    if 'email' in data:
        email_text = data['email']
    elif 'content' in data:
        email_text = data['content']
    
    if not email_text:
        return jsonify({'error': 'Missing email content'}), 400
    
    # Make prediction
    try:
        result = predict_email(email_text)
        
        # Map result to ham/spam if needed (depending on your model's output labels)
        # Assuming the model's labels are already 'ham'/'spam' or 'legitimate'/'phishing'
        if result.lower() in ['legitimate', 'ham']:
            classification = 'ham'
            stats['legitimate_emails'] += 1
        else:
            classification = 'spam'
            stats['phishing_emails'] += 1
        
        # Update stats
        stats['total_emails'] += 1
        if stats['total_emails'] > 0:
            phishing_percentage = (stats['phishing_emails'] / stats['total_emails']) * 100
            stats['detection_rate'] = f"{phishing_percentage:.1f}%"
        
        # Add to recent detections (keep only latest 10)
        stats['recent_detections'].insert(0, {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'result': classification,
            'email_preview': email_text[:50] + '...' if len(email_text) > 50 else email_text
        })
        stats['recent_detections'] = stats['recent_detections'][:10]
        
        # Return standardized response format
        return jsonify({
            'success': True,
            'result': classification,
            'original_label': result,
            'confidence': 0.9,  # Placeholder until we implement confidence scores
            'email_preview': email_text[:100] + '...' if len(email_text) > 100 else email_text
        })
    
    except Exception as e:
        print(f"Error analyzing email: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with more system information"""
    uptime = (datetime.datetime.now() - datetime.datetime.fromisoformat(stats['start_time'])).total_seconds()
    return jsonify({
        'status': 'healthy' if MODEL_LOADED else 'degraded',
        'model_loaded': MODEL_LOADED,
        'url_model_loaded': url_model is not None,
        'api_version': stats['api_version'],
        'uptime_seconds': uptime,
        'cache_size': len(url_cache),
        'total_scans': stats['total_emails'] + stats['url_scans']
    })

# Statistics endpoint for the extension
@app.route('/stats', methods=['GET'])
def api_stats():
    return jsonify({
        'total_emails': stats['total_emails'],
        'phishing_emails': stats['phishing_emails'],
        'legitimate_emails': stats['legitimate_emails'],
        'detection_rate': stats['detection_rate'],
        'recent_count': len(stats['recent_detections']),
        'url_scans': stats['url_scans'],
        'malicious_urls': stats['malicious_urls'],
        'url_detection_rate': f"{(stats['malicious_urls'] / stats['url_scans'] * 100):.1f}%" if stats['url_scans'] > 0 else '0%',
        'cache_hit_rate': stats.get('cache_hits', 0) / stats['url_scans'] if stats['url_scans'] > 0 else 0
    })

### URL SCANNER MODEL SETUP ###
url_model = None

def get_url_model():
    global url_model
    if url_model is None:
        # Construct absolute path to URL classifier
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'malicious_url_model', 'url_classifier_model.pkl'))
        try:
            with open(model_path, 'rb') as f:
                url_model = pickle.load(f)
            print(f"URL model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading URL model: {e}")
            # Try backup location
            backup_path = os.path.join(os.path.dirname(__file__), 'url_classifier_model.pkl')
            if os.path.exists(backup_path):
                with open(backup_path, 'rb') as f:
                    url_model = pickle.load(f)
                print(f"URL model loaded from backup location: {backup_path}")
    return url_model

def normalize_url(url):
    """Normalize URL for consistent caching and processing"""
    if not url:
        return ""
    # Convert to lowercase
    url = url.lower().strip()
    # Remove common prefixes if present
    if url.startswith(('http://', 'https://', 'ftp://')):
        return url
    # Add https:// prefix if missing
    return f"https://{url}" if not url.startswith('http') else url

def scan_single_url(url):
    """Scan a single URL and determine if it's malicious"""
    # Check cache first
    normalized_url = normalize_url(url)
    
    with URL_CACHE_LOCK:
        if normalized_url in url_cache:
            # Move to end to mark as recently used
            result = url_cache.pop(normalized_url)
            url_cache[normalized_url] = result
            # Track cache hits for statistics
            stats['cache_hits'] = stats.get('cache_hits', 0) + 1
            return result
    
    try:
        model = get_url_model()
        if not model:
            raise ValueError("URL scanning model not available")
            
        pred = model.predict([normalized_url])[0]
        
        # Map prediction to a standard result
        if isinstance(pred, str):
            result = pred
        else:
            result = 'unsafe' if pred else 'safe'
        
        is_malicious = str(result).lower() in ['unsafe', 'spam', 'phishing']
        
        # Create result object
        scan_result = {
            'url': url,
            'isMalicious': is_malicious,
            'rawResult': result,
            'timestamp': time.time()
        }
        
        # Update cache
        with URL_CACHE_LOCK:
            url_cache[normalized_url] = scan_result
            # Ensure cache doesn't grow too large
            if len(url_cache) > URL_CACHE_SIZE:
                url_cache.popitem(last=False)  # Remove oldest item
        
        # Update stats
        stats['url_scans'] += 1
        if is_malicious:
            stats['malicious_urls'] += 1
            
        return scan_result
        
    except Exception as e:
        print(f"Error scanning URL {url}: {e}")
        return {
            'url': url,
            'error': str(e),
            'isMalicious': False,
            'rawResult': 'error'
        }

# New endpoint to scan a URL
@app.route('/scan-url', methods=['POST'])
@app.route('/api/scan/url', methods=['POST'])  # New standardized endpoint
def scan_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing url parameter'}), 400
        
    url = data['url']
    result = scan_single_url(url)
    return jsonify(result)

# Optimized batch URL scanning
@app.route('/api/scan/urls', methods=['POST'])
def scan_urls_batch():
    """Endpoint to scan multiple URLs in a single request"""
    data = request.get_json()
    if not data or 'urls' not in data or not isinstance(data['urls'], list):
        return jsonify({'error': 'Missing or invalid urls parameter'}), 400
    
    urls = data['urls']
    if not urls:
        return jsonify({'results': []})
    
    # Limit batch size to a reasonable number
    max_batch = 100  # Process at most 100 URLs per request
    urls = urls[:max_batch]
    
    print(f"Batch scanning {len(urls)} URLs")
    
    # Check which URLs are already in cache
    cached_results = {}
    urls_to_scan = []
    normalized_urls = []
    
    # First check cache for all URLs
    with URL_CACHE_LOCK:
        for url in urls:
            normalized_url = normalize_url(url)
            normalized_urls.append(normalized_url)
            
            if normalized_url in url_cache:
                # Use cached result
                cached_results[url] = url_cache[normalized_url]
                # Move to end of cache (mark as recently used)
                url_cache.move_to_end(normalized_url)
                # Track cache hit
                stats['cache_hits'] = stats.get('cache_hits', 0) + 1
            else:
                urls_to_scan.append((url, normalized_url))
    
    # Get URL model for batch scanning
    model = get_url_model() if urls_to_scan else None
    results = []
    
    # Add all cached results to the results list
    for url, result in cached_results.items():
        results.append(result)
    
    # Process non-cached URLs in a batch if possible
    if urls_to_scan and model:
        try:
            # Prepare list of normalized URLs for batch prediction
            batch_urls = [pair[1] for pair in urls_to_scan]
            # Make batch prediction
            preds = model.predict(batch_urls)
            
            # Process predictions
            for i, (url, normalized_url) in enumerate(urls_to_scan):
                pred = preds[i]
                
                # Map prediction to standard result
                if isinstance(pred, str):
                    result = pred
                else:
                    result = 'unsafe' if pred else 'safe'
                
                is_malicious = str(result).lower() in ['unsafe', 'spam', 'phishing']
                
                # Create result object
                scan_result = {
                    'url': url,
                    'isMalicious': is_malicious,
                    'rawResult': result,
                    'timestamp': time.time()
                }
                
                # Add to results list
                results.append(scan_result)
                
                # Update cache
                with URL_CACHE_LOCK:
                    url_cache[normalized_url] = scan_result
                    # Ensure cache doesn't grow too large
                    if len(url_cache) > URL_CACHE_SIZE:
                        url_cache.popitem(last=False)
                
                # Update stats
                stats['url_scans'] += 1
                if is_malicious:
                    stats['malicious_urls'] += 1
                
        except Exception as e:
            print(f"Error in batch URL scanning: {e}")
            # If batch fails, try scanning URLs individually
            for url, normalized_url in urls_to_scan:
                result = scan_single_url(url)
                results.append(result)
    
    return jsonify({'results': results})

# For backward compatibility
@app.route('/scan-urls-batch', methods=['POST'])
def scan_urls_batch_legacy():
    """Legacy endpoint for batch URL scanning (redirects to new endpoint)"""
    return scan_urls_batch()

# Endpoint to get scanning statistics
@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Endpoint to get detailed API metrics for monitoring"""
    uptime = (datetime.datetime.now() - datetime.datetime.fromisoformat(stats['start_time'])).total_seconds()
    
    return jsonify({
        'uptime_seconds': uptime,
        'api_version': stats['api_version'],
        'url_scans': {
            'total': stats['url_scans'],
            'malicious': stats['malicious_urls'],
            'detection_rate': f"{(stats['malicious_urls'] / stats['url_scans'] * 100):.1f}%" if stats['url_scans'] > 0 else '0%',
            'cache_hit_rate': stats.get('cache_hits', 0) / stats['url_scans'] if stats['url_scans'] > 0 else 0,
            'cache_size': len(url_cache),
            'cache_max_size': URL_CACHE_SIZE
        },
        'email_scans': {
            'total': stats['total_emails'],
            'phishing': stats['phishing_emails'],
            'legitimate': stats['legitimate_emails'],
            'detection_rate': stats['detection_rate']
        },
        'models': {
            'email_model_loaded': MODEL_LOADED,
            'url_model_loaded': url_model is not None
        },
        'recent_scans': stats['recent_detections'][:5]
    })

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 