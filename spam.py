 # -----------------------------
# 1️⃣  Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# -----------------------------
# 2️⃣  Load Dataset
# -----------------------------
# If using local file:
df = pd.read_csv('H:/python Internship/NLPmodel/spamm.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())

# -----------------------------
# 3️⃣  EDA (Exploratory Data Analysis)
# -----------------------------
sns.countplot(x='label', data=df)
plt.title('Spam vs Ham Count')
plt.show()

# -----------------------------
# 4️⃣  Text Preprocessing
# -----------------------------
def clean_text(msg):
    msg = "".join([char for char in msg if char not in string.punctuation])
    words = msg.split()
    stopwords_list = stopwords.words('english')
    return " ".join([word.lower() for word in words if word.lower() not in stopwords_list])

df['cleaned_message'] = df['message'].apply(clean_text)

# -----------------------------
# 5️⃣  Feature Extraction
# -----------------------------
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned_message'])
y = df['label']

# -----------------------------
# 6️⃣  Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 7️⃣  Model Training
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 8️⃣  Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------

#Evaluate
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# # 9️⃣  Evaluation Metrics
# # -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# -----------------------------
# 1️⃣0️⃣  Visualize Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Example: use the clean spam dataset
# import pandas as pd

# df = pd.read_csv('H:/python Internship/NLPmodel/spamm.csv', encoding='latin-1')

# # Rename columns if needed
# df.columns = ['label', 'message']

# # Convert labels to binary: spam=1, ham=0
# df['label_num'] = df.label.map({'ham':0, 'spam':1})

# # Split features and labels
# X = df['message']
# y = df['label_num']

# # Vectorize text to bag-of-words
# vectorizer = CountVectorizer()
# X_vectorized = vectorizer.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_vectorized, y, test_size=0.2, random_state=42
# )

# # Train a simple Naive Bayes classifier
# model = MultinomialNB()
# model.fit(X_train, y_train)

# # Predict on test set
# y_pred = model.predict(X_test)

# # Evaluate
# accuracy = accuracy_score(y_test, y_pred) * 100
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f"✅ Accuracy: {accuracy:.4f}")
# print(f"✅ Precision: {precision:.4f}")
# print(f"✅ Recall: {recall:.4f}")
# print(f"✅ F1 Score: {f1:.4f}")
