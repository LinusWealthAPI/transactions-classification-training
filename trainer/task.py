import re
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import pytz
from sklearn.preprocessing import MaxAbsScaler
from dotenv import load_dotenv


load_dotenv()

# Load training data
training_data_path = os.getenv('TRAINING_DATA_PATH')
df = pd.read_csv(training_data_path)

# Preprocess 
df['counterpart_name'] = df['counterpart_name'].fillna('')
df['purpose'] = df['purpose'].fillna('')

df["data"] = df["counterpart_name"] + " " + df["purpose"]
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

df["counterpart_name"] = df["counterpart_name"].apply(remove_numbers)
df["purpose"] = df["purpose"].apply(remove_numbers)

# Select training data
X = df[["counterpart_name", "purpose"]]
y = df["category"]

# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model Pipeline
num_features_to_select = 8000  # Adjust this number based on your dataset and experimentation

# Column Transformer to apply separate transformations to each column
preprocessor = ColumnTransformer(
    transformers=[
        ('counterpart_name', CountVectorizer(), 'counterpart_name'),
        ('purpose', CountVectorizer(), 'purpose')
    ],
    remainder='drop'
)

# Create model Pipeline with manual feature selection and adjusted regularization strength
model = Pipeline([
    ('preprocessor', preprocessor),
    ('select_k_best', SelectKBest(score_func=chi2, k=num_features_to_select)),
    ('scaler', MaxAbsScaler()),
    ('clf', LogisticRegression(penalty='l2', C=0.2, max_iter=1000))
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

output_location = os.getenv('OUTPUT_LOCATION')
directory_path = os.path.join(output_location, f"model_{start_time.strftime('%d.%m.%Y_%H:%M:%S')}")

os.makedirs(directory_path)

joblib.dump(model, f'{directory_path}/trained_model.joblib')

with open(f'{directory_path}/model_evaluation.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy:.2f}\n\n')
    file.write(f"Time elapsed: {time_delta}\n")
    file.write('Classification Report:\n')
    file.write(classification_rep)