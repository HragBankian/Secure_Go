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
import json
import traceback
from datetime import timedelta

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

# Import DB handler early to ensure we can load stats before defining the stats dictionary
try:
    from db_handler import save_stats, get_historical_stats, load_last_stats, DB_AVAILABLE
    print("Database handler imported successfully")
except ImportError as e:
    print(f"Warning: Could not import database handler: {e}")
    DB_AVAILABLE = False
    # Create dummy functions in case the import fails
    def save_stats(stats):
        print("Database functionality not available - stats not saved")
        return False
    def get_historical_stats(days=7):
        return []
    def load_last_stats():
        return {}

# Try to load previous stats from database
previous_stats = {}
if DB_AVAILABLE:
    try:
        previous_stats = load_last_stats()
        print(f"Loaded previous stats: Emails={previous_stats.get('total_emails', 0)}, URLs={previous_stats.get('url_scans', 0)}")
    except Exception as e:
        print(f"Error loading previous stats: {e}")
        previous_stats = {}

# Track some basic stats, initialized with previous values if available
stats = {
    'total_emails': previous_stats.get('total_emails', 0),
    'phishing_emails': previous_stats.get('phishing_emails', 0),
    'legitimate_emails': previous_stats.get('legitimate_emails', 0),
    'detection_rate': previous_stats.get('detection_rate', '0%'),
    'recent_detections': previous_stats.get('recent_detections', []),
    'url_scans': previous_stats.get('url_scans', 0),
    'malicious_urls': previous_stats.get('malicious_urls', 0),
    'start_time': datetime.datetime.now().isoformat(),
    'api_version': '1.1.0',  # Track API version
    'last_db_save': 'Never'
}

print(f"Initial stats: {stats['total_emails']} emails, {stats['url_scans']} URLs scanned")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Cache for URL scanning results - OrderedDict works as LRU cache
url_cache: OrderedDict = OrderedDict()
URL_CACHE_SIZE = 10000  # Maximum number of URLs to cache
URL_CACHE_LOCK = threading.Lock()  # Thread safety for cache access

# Create a save lock and tracking variables for optimized saving
SAVE_LOCK = threading.Lock()
last_save_time = time.time()
pending_changes = False
save_timer = None
MIN_SAVE_INTERVAL = 10  # Minimum seconds between saves (to prevent excessive saves)

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
    # Add TfidfVectorizer to PyTorch's safe globals list to fix the loading issue in PyTorch 2.6+
    try:
        import sklearn.feature_extraction.text
        import torch.serialization
        torch.serialization.add_safe_globals([sklearn.feature_extraction.text.TfidfVectorizer])
        print("Added TfidfVectorizer to safe globals list")
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not add TfidfVectorizer to safe globals: {e}")
        
    try:
        # Try with the new safe globals approach first
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"First load attempt failed: {e}")
        # Fall back to the legacy approach (less secure but needed for older models)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        print("Loaded model with weights_only=False (legacy mode)")
    
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

# Function to efficiently handle saving stats to MongoDB
def save_stats_if_needed(force=False):
    """
    Save stats to MongoDB only if necessary and not too frequent.
    This reduces database load while ensuring stats are still saved.
    
    Args:
        force: If True, save regardless of timing constraints
    """
    global last_save_time, pending_changes, save_timer
    
    # If database isn't available, don't bother
    if not DB_AVAILABLE:
        return False
        
    current_time = time.time()
    time_since_last_save = current_time - last_save_time
    
    # Only acquire the lock if we might actually save
    if force or (pending_changes and time_since_last_save >= MIN_SAVE_INTERVAL):
        with SAVE_LOCK:
            # Check again inside the lock
            if force or (pending_changes and time_since_last_save >= MIN_SAVE_INTERVAL):
                try:
                    # Perform the actual save
                    print(f"Saving stats to MongoDB (forced={force})")
                    save_success = save_stats(stats)
                    
                    if save_success:
                        # Update tracking variables
                        last_save_time = time.time()
                        pending_changes = False
                        stats['last_db_save'] = datetime.datetime.now().isoformat()
                        print(f"Stats saved: {stats['total_emails']} emails, {stats['url_scans']} URLs")
                        return True
                    else:
                        print("Failed to save stats to MongoDB")
                except Exception as e:
                    print(f"Error saving stats to MongoDB: {e}")
    
    # If we didn't save but have pending changes, schedule a future save
    if pending_changes and save_timer is None:
        # Calculate how long to wait before the next save
        wait_time = max(0, MIN_SAVE_INTERVAL - time_since_last_save)
        save_timer = threading.Timer(wait_time, delayed_save)
        save_timer.daemon = True  # Allow the program to exit if this is still running
        save_timer.start()
        print(f"Scheduled save in {wait_time:.1f} seconds")
    
    return False

def delayed_save():
    """Handler for the delayed save timer"""
    global save_timer
    save_timer = None  # Clear the timer reference
    save_stats_if_needed(force=True)  # Force a save

def mark_stats_changed():
    """
    Mark that stats have changed and need to be saved
    """
    global pending_changes
    pending_changes = True
    # Try to save if it's been long enough
    save_stats_if_needed()

# Replace the analyze_email function with this optimized version
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
        
        # Mark stats as changed - this will trigger a save when appropriate
        mark_stats_changed()
        
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
        
        # Make sure we handle pickle loading safely
        try:
            try:
                # First attempt: Use pickle with default safety
                with open(model_path, 'rb') as f:
                    url_model = pickle.load(f)
                print(f"URL model loaded from {model_path}")
            except Exception as primary_error:
                print(f"Primary load attempt failed: {primary_error}")
                # Try backup location
                backup_path = os.path.join(os.path.dirname(__file__), 'url_classifier_model.pkl')
                if os.path.exists(backup_path):
                    with open(backup_path, 'rb') as f:
                        url_model = pickle.load(f)
                    print(f"URL model loaded from backup location: {backup_path}")
                else:
                    raise FileNotFoundError(f"Could not find URL model at either path")
        except Exception as e:
            print(f"Error loading URL model: {e}")
            print("URL scanning functionality will be unavailable")
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

# Replace the scan_url function with this optimized version
@app.route('/scan-url', methods=['POST'])
@app.route('/api/scan/url', methods=['POST'])  # New standardized endpoint
def scan_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'Missing url parameter'}), 400
        
    url = data['url']
    result = scan_single_url(url)
    
    # Mark stats as changed for efficient saving
    mark_stats_changed()
    
    return jsonify(result)

# Replace the batch URL scanning function with an optimized version
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
    stats_changed = False  # Track if any stats were updated
    
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
                stats_changed = True
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
                stats_changed = True
                
        except Exception as e:
            print(f"Error in batch URL scanning: {e}")
            # If batch fails, try scanning URLs individually
            for url, normalized_url in urls_to_scan:
                result = scan_single_url(url)
                results.append(result)
                stats_changed = True
    
    # Mark stats as changed if we updated any stats
    if stats_changed:
        mark_stats_changed()
    
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
        'recent_scans': stats['recent_detections'][:5],
        'database': {
            'available': DB_AVAILABLE,
            'last_save_attempt': stats['last_db_save']
        }
    })

# Set up a periodic task to save stats to the database (if available)
def save_stats_periodically():
    """Save the current stats to the database on a schedule"""
    # Force a save regardless of the timer constraints
    save_stats_if_needed(force=True)
    
    # Schedule the next periodic save (every 5 minutes)
    threading.Timer(5 * 60, save_stats_periodically).start()

# Update the refresh stats endpoint to use the new saving mechanism
@app.route('/api/db/refresh', methods=['POST'])
def refresh_stats():
    """Force an immediate save and reload of stats"""
    if not DB_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "Database functionality not available"
        }), 503
    
    global stats
    
    try:
        # First save current stats (force=True ensures immediate save)
        print("Forcing immediate save to database...")
        save_success = save_stats_if_needed(force=True)
        
        if save_success:
            # Then reload from database to verify they're retrievable
            print("Reloading stats from database...")
            loaded_stats = load_last_stats()
            
            if loaded_stats:
                # Keep the current start time and API version
                loaded_stats['start_time'] = stats['start_time']
                loaded_stats['api_version'] = stats['api_version']
                
                # Update the stats object with loaded values
                stats = loaded_stats
                print(f"Stats refreshed: {stats['total_emails']} emails, {stats['url_scans']} URLs")
                
                return jsonify({
                    "success": True,
                    "message": "Stats saved and reloaded successfully",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "stats": {
                        "total_emails": stats['total_emails'],
                        "phishing_emails": stats['phishing_emails'],
                        "legitimate_emails": stats['legitimate_emails'],
                        "url_scans": stats['url_scans'],
                        "malicious_urls": stats['malicious_urls']
                    }
                })
            else:
                return jsonify({
                    "success": True,
                    "message": "Stats saved but no previous stats were found to load",
                    "timestamp": datetime.datetime.now().isoformat()
                })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to save stats to database",
                "timestamp": datetime.datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Add an endpoint for retrieving stats from the database
@app.route('/api/db/stats', methods=['GET'])
def get_db_stats():
    """Get historical statistics from the database"""
    try:
        # Get the 'days' parameter from the request
        days = request.args.get('days', default=7, type=int)
        
        # Get the historical stats from the database
        historical_stats = get_historical_stats(days)
        
        # Add the current stats
        current_stats = {
            "timestamp": datetime.datetime.now().isoformat(),
            "email_stats": {
                "total_emails": stats['total_emails'],
                "phishing_emails": stats['phishing_emails'],
                "legitimate_emails": stats['legitimate_emails'],
                "detection_rate": stats['detection_rate']
            },
            "url_stats": {
                "total_urls": stats['url_scans'],
                "malicious_urls": stats['malicious_urls']
            },
            "api_version": stats['api_version']
        }
        
        return jsonify({
            "current": current_stats,
            "historical": historical_stats,
            "db_available": DB_AVAILABLE
        })
    
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error retrieving DB stats: {e}\n{traceback_str}")
        return jsonify({
            "error": str(e),
            "db_available": DB_AVAILABLE
        }), 500

# Update the trigger_save_stats endpoint to use the new saving mechanism
@app.route('/api/db/save-stats', methods=['POST'])
def trigger_save_stats():
    """Manually trigger saving the current stats to the database"""
    if not DB_AVAILABLE:
        return jsonify({
            "success": False,
            "message": "Database functionality not available"
        }), 503
    
    try:
        success = save_stats_if_needed(force=True)
        return jsonify({
            "success": success,
            "message": "Stats saved successfully" if success else "Failed to save stats",
            "timestamp": datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Initialize periodic save if database is available
    if DB_AVAILABLE:
        print("Starting enhanced stats saving system...")
        # Save immediately once, then schedule future saves
        try:
            initial_save = save_stats_if_needed(force=True)
            print(f"Initial stats save {'succeeded' if initial_save else 'failed'}")
            if initial_save:
                print(f"Initial stats saved: {stats['total_emails']} emails, {stats['url_scans']} URLs")
        except Exception as e:
            print(f"Error during initial stats save: {e}")
        
        # Start the timer for periodic saves - every 5 minutes
        print("Scheduling periodic saves every 5 minutes...")
        threading.Timer(5 * 60, save_stats_periodically).start()
    else:
        print("WARNING: Database functionality not available - statistics will not be persisted")
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 