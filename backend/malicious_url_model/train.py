import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Paths
DATASET_PATH = 'synthetic_urls.csv'
MODEL_OUTPUT_PATH = 'url_classifier_model.pkl'

# Load the synthetic dataset
def load_data():
    df = pd.read_csv(DATASET_PATH)
    return df

# Save the model to a file
def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to '{path}'")

# Train the model
def train_model(df):
    X = df['url']
    y = df['label']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build a pipeline: TF-IDF -> Random Forest
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 3))),
        ('clf', RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    # Train
    print("Training the model...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds.")

    # Evaluate
    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return pipeline

# Main function
def main():
    print("Loading data...")
    df = load_data()
    print(f"Dataset loaded: {df.shape[0]} samples")

    print("Starting training process...")
    model = train_model(df)

    print("Saving model...")
    save_model(model, MODEL_OUTPUT_PATH)

if __name__ == "__main__":
    main()
