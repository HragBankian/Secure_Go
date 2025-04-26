import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
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

# Custom Dataset class
class EmailDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), self.y[idx]

# Neural Network model
class PhishingDetectionModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super(PhishingDetectionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def main():
    print("Loading and preprocessing data...")
    
    # Load the data
    try:
        # Try to load both CSV files and concatenate
        df1 = pd.read_csv('backend_gev/mail_data_1.csv')
        df2 = pd.read_csv('backend_gev/mail_data_2.csv')
        df = pd.concat([df1, df2], ignore_index=True)
    except Exception as e:
        print(f"Error loading one of the files: {e}")
        # Fall back to just the first file
        df = pd.read_csv('backend_gev/mail_data_1.csv')
    
    # Let's examine the dataframe structure
    print(f"Dataframe columns: {df.columns.tolist()}")
    
    # Assuming the dataframe has 'text' and 'label' columns
    # If the columns have different names, adjust accordingly
    text_column = 'text' if 'text' in df.columns else df.columns[0]
    label_column = 'label' if 'label' in df.columns else df.columns[1]
    
    print(f"Using text column: {text_column}")
    print(f"Using label column: {label_column}")
    
    # Preprocess the text
    print("Preprocessing emails...")
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Check value counts of labels
    print(f"Label distribution: {df[label_column].value_counts()}")
    
    # Data cleaning: Make sure all values in label_column are valid
    # Keep only rows with non-null values in both columns
    df = df.dropna(subset=[text_column, label_column])
    
    # Convert labels to binary classes (0 and 1) for simplicity
    # This helps avoid CUDA errors with invalid class indices
    print("Encoding labels...")
    df[label_column] = df[label_column].astype(str)
    
    # Get unique labels and print them
    unique_labels = df[label_column].unique()
    print(f"Unique labels before encoding: {unique_labels}")
    
    # Convert labels to numeric if needed
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_column])
    print(f"Encoded labels: {label_encoder.classes_}")
    print(f"Transformed label values: {np.unique(y)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Feature extraction with TF-IDF
    print("Extracting features with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['processed_text']).toarray()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Convert to PyTorch datasets
    train_dataset = EmailDataset(X_train, torch.LongTensor(y_train))
    test_dataset = EmailDataset(X_test, torch.LongTensor(y_test))
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = PhishingDetectionModel(input_size, num_classes=len(label_encoder.classes_))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("Training the model...")
    num_epochs = 10
    
    # Force CPU training to avoid CUDA errors
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    # Evaluation
    print("Evaluating the model...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Save the model and vectorizer
    print("Saving the model...")
    model_path = 'backend_gev/phishing_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'vectorizer': vectorizer,
        'label_encoder': label_encoder
    }, model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 