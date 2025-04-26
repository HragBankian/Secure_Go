import pickle
import time

# Path to the saved model
MODEL_PATH = 'url_classifier_model.pkl'

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
