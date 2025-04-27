import pickle
import time

# Path to the saved model
MODEL_PATH = 'backend/malicious_url_model/url_classifier_model.pkl'

# Specifically excluded URLs that should never be flagged
EXCLUDED_URLS = [
    '10.170.8.90:5000',
    '10.170.8.90'
]

# Load the trained model
def load_model(path):
    start = time.time()
    with open(path, 'rb') as file:
        model = pickle.load(file)
    end = time.time()
    print(f"Model loaded in {end - start:.2f} seconds.")
    return model

# Predict the label for a given URL
def predict_url(model, url):
    # Check for excluded URLs first
    for excluded in EXCLUDED_URLS:
        if excluded in url:
            print(f"URL contains excluded domain {excluded}, automatically classifying as safe.")
            return 'safe'
    
    start = time.time()
    prediction = model.predict([url])[0]
    end = time.time()
    print(f"Prediction made in {end - start:.4f} seconds.")
    return prediction

# Main execution
def main():
    model = load_model(MODEL_PATH)

    while True:
        user_input = input("\nEnter a URL to classify (or type 'exit' to quit): ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting...")
            break

        prediction = predict_url(model, user_input)
        print(f"The URL is classified as: {prediction}")

if __name__ == "__main__":
    main()
