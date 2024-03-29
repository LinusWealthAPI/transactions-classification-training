import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import pytz

# Load training data
df = pd.read_csv("/gcs/transaction_classification_frankfurt/training_data/data.csv")

# Preprocess 
df['counterpart_name'] = df['counterpart_name'].fillna('')
df['purpose'] = df['purpose'].fillna('')

df["data"] = df["counterpart_name"] + " " + df["purpose"]
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

df["data"] = df["data"].apply(remove_numbers)

# Select training data
X = df["data"]
y = df["category"]

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model Pipeline
model = Pipeline([
    ('count_vectorizer', CountVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Fit training data
start_time = datetime.now(tz=pytz.timezone("Europe/Berlin"))
print(f"Start training at {start_time}")
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:\n', classification_rep)

# Save evaluation and model
end_time = datetime.now(tz=pytz.timezone("Europe/Berlin"))

time_delta = end_time - start_time

directory_path = f"/gcs/transaction_classification_frankfurt/training_output/model_{start_time.strftime('%d.%m.%Y_%H:%M:%S')}"

os.makedirs(directory_path)

joblib.dump(model, f'{directory_path}/trained_model.joblib')

with open(f'{directory_path}/model_evaluation.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy:.2f}\n\n')
    file.write(f"Time elapsed: {time_delta}\n")
    file.write('Classification Report:\n')
    file.write(classification_rep)