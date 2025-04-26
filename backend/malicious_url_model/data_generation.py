import random
import string
import pandas as pd

# Define lists of domains and keywords for different types of URLs
safe_domains = ['google.com', 'yahoo.com', 'facebook.com', 'amazon.com', 'twitter.com']
unsafe_keywords = ['login', 'secure', 'account', 'banking', 'download', 'torrent', 'exe', 'virus', 'free', 'update']

# Function to generate random strings for unsafe URLs
def random_string(length=10):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

# Function to generate safe URLs (legitimate domains with random paths)
def generate_safe_url():
    domain = random.choice(safe_domains)
    path = random_string(random.randint(5, 15))  # Path of varying lengths
    return f"https://{domain}/{path}"

# Function to generate unsafe URLs (phishing, malware, etc.)
def generate_unsafe_url():
    # Randomly choose between phishing, malware, and defacement patterns
    label = random.choice(['phishing', 'malware', 'defacement'])
    
    if label == 'phishing':
        domain = f"{random.choice(unsafe_keywords)}-{random_string(5)}.com"
        path = f"/{random.choice(unsafe_keywords)}-{random_string(10)}"
    elif label == 'malware':
        domain = f"malicious-{random_string(6)}.com"
        path = f"/{random.choice(unsafe_keywords)}/{random_string(10)}"
    else:  # Defacement
        domain = f"vulnerable-{random_string(5)}.com"
        path = f"/{random.choice(unsafe_keywords)}"
    
    return f"https://{domain}{path}"

# Function to generate synthetic data (100,000 URLs)
def generate_data(num_samples=100000):
    data = []
    
    for _ in range(num_samples):
        # Randomly choose between safe and unsafe
        if random.choice([True, False]):
            url = generate_safe_url()
            label = 'safe'
        else:
            url = generate_unsafe_url()
            label = 'unsafe'
        
        data.append({'url': url, 'label': label})
    
    return pd.DataFrame(data)

# Generate the dataset
synthetic_data = generate_data()

# Save the dataset to a CSV file
synthetic_data.to_csv('synthetic_urls.csv', index=False)

print(f"Data generation complete. 100,000 synthetic URLs generated and saved to 'synthetic_urls.csv'")
