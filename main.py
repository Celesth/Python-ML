import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample data
texts = [
    "I love this product",
    "This is the best thing ever",
    "I am so happy with this",
    "I hate this product",
    "This is the worst thing ever",
    "I am so unhappy with this"
]

# Corresponding labels (1 for positive, 0 for negative)
labels = [1, 1, 1, 0, 0, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Convert labels to a NumPy array
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

new_texts = ["I am very happy with this service", "This is a terrible experience"]
new_X = vectorizer.transform(new_texts)
predictions = model.predict(new_X)
print(predictions)  # Output will be an array of predicted labels