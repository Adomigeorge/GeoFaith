# Import libraries for data manipulation and analysis
import pandas as pd  # For handling datasets
import numpy as np   # For numerical computations

# Import libraries for natural language processing (NLP)
import re  # For regular expressions to clean text
import nltk  # For text processing
from nltk.corpus import stopwords  # To remove common, non-informative words
from nltk.tokenize import word_tokenize  # For splitting text into words
from nltk.stem import PorterStemmer  # For stemming (reducing words to their root form)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # For text vectorization

# Import libraries for machine learning
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # Random forest algorithm for classification
from sklearn.linear_model import LogisticRegression  # Logistic regression for binary classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # To evaluate model performance

# Import libraries for data visualization
import matplotlib.pyplot as plt  # For creating plots and graphs
import seaborn as sns  # For more advanced and visually appealing plots

# Handle warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

%clear #clear console

# Load the dataset from a CSV file
data = pd.read_csv('C:/Users/Admin/Desktop/georges/New folder/mental_health.csv')  # Replace 'your_dataset.csv' with the actual file path

# Display the first few rows of the dataset
print(data.head())

# Check the structure of your dataset (columns, data types, etc.)
print(data.info())

# Check for missing values
print(data.isnull().sum())  # Shows the number of missing values in each column

# Select only the relevant columns
text_column = 'post_text'  # Replace with the name of your text column
label_column = 'label'      # Replace with the name of your label column

# Extract text and label columns
texts = data[text_column]
labels = data[label_column]
print(texts.head())  # Preview the text data
print(labels.head())  # Preview the labels

# Define a function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'\bwww\.\S+|http[s]?://\S+', '', text)
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply the cleaning function
texts = texts.apply(clean_text)
print(texts.head())  # Check cleaned text

# Tokenize each cleaned text
tokenized_texts = texts.apply(word_tokenize)
print(tokenized_texts.head())  # Preview tokenized text

nltk.download('punkt')  # Required for word_tokenize
nltk.download('stopwords')

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords from the tokenized text
processed_texts = tokenized_texts.apply(lambda words: [word for word in words if word.lower() not in stop_words])
print(processed_texts.head())  # Preview text without stopwords

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  # Use top 1000 features and exclude stopwords

# Apply the vectorizer to the cleaned and processed text
tfidf_features = tfidf_vectorizer.fit_transform(texts)

# Convert the features into a DataFrame for easier exploration
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=feature_names)

# Display a preview of the TF-IDF features
print(tfidf_df.head())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_features, labels, test_size=0.2, random_state=42
)

# Check the shape of the splits
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Initialize the model
logistic_model = LogisticRegression(random_state=42)

# Train the model on the training data
logistic_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10],  # Inverse regularization strength
    'solver': ['liblinear', 'saga'],  # Optimization algorithms
    'penalty': ['l1', 'l2']  # Regularization penalties
}

# Initialize Grid Search with Logistic Regression
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')

# Perform the search on training data
grid_search.fit(X_train, y_train)

# Retrieve the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model to predict on test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the tuned model
print("Tuned Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = random_forest_model.predict(X_test)

# Evaluate the model's performance
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

import joblib
joblib.dump(random_forest_model, 'random_forest_model.pkl')  # Save the model
loaded_model = joblib.load('random_forest_model.pkl')  # Load the model later

#Interpretation:
feature_importances = random_forest_model.feature_importances_
important_features = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
print(important_features.head(10))  # Top 10 most important features

# Example of new data (replace 'new_texts' with actual input)
new_texts = ["Feeling very happy today"]
new_features = tfidf_vectorizer.transform(new_texts)  # Convert the text to TF-IDF features
predictions = loaded_model.predict(new_features)     # Predict with the loaded model
print(predictions)  # Output: [1, 0] (1 = depressive, 0 = non-depressive)
