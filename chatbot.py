"""
import nltk
from nltk.tokenize import word_tokenize
import random

nltk.download('punkt')
nltk.download('punkt_tab')

print("🤖 Chatbot: Hello! Type 'quit' to exit.")

# Intent responses
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help you?"],
    "goodbye": ["Goodbye!", "See you later!", "Bye! Have a great day!"],
    "name": ["I'm a chatbot created by Chandru.", "You can call me AI Buddy."],
    "help": ["I can answer simple questions. Try greeting me!"]
}

def detect_intent(user_input):
    tokens = word_tokenize(user_input.lower())

    if any(word in tokens for word in ["hi", "hello", "hey"]):
        return "greeting"
    elif any(word in tokens for word in ["bye", "goodbye", "see you"]):
        return "goodbye"
    elif any(word in tokens for word in ["name", "who"]):
        return "name"
    elif any(word in tokens for word in ["help", "assist"]):
        return "help"
    else:
        return "unknown"

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("🤖 Chatbot: Goodbye!")
        break

    intent = detect_intent(user_input)

    if intent in responses:
        print("🤖 Chatbot:", random.choice(responses[intent]))
    else:
        print("🤖 Chatbot: I'm not sure how to respond to that.")
"""
"""

import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from intents import intents

print("🤖 Chatbot: Hello! Type 'quit' to exit.")

# ----------------------------
# 1. Prepare training data
# ----------------------------
sentences = []
labels = []

for intent, examples in intents.items():
    for example in examples:
        sentences.append(example)
        labels.append(intent)

# ----------------------------
# 2. Convert text to numbers
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

# ----------------------------
# 3. Train classifier
# ----------------------------
model = LogisticRegression()
model.fit(X, labels)

# ----------------------------
# 4. Chat loop
# ----------------------------
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!"],
    "goodbye": ["Goodbye!", "See you later!"],
    "name": ["I'm your ML chatbot.", "You can call me AI Buddy."],
    "help": ["I can chat and answer simple questions."]
}

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("🤖 Chatbot: Goodbye!")
        break

    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)[0]

    if prediction in responses:
        print("🤖 Chatbot:", random.choice(responses[prediction]))
    else:
        print("🤖 Chatbot: I'm not sure how to respond to that.")


"""

import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from intents import intents

print("🤖 Chatbot: Hello! Type 'quit' to exit.")

# ----------------------------
# 1. Prepare training data
# ----------------------------
sentences = []
labels = []

for intent, examples in intents.items():
    for example in examples:
        sentences.append(example)
        labels.append(intent)

# ----------------------------
# 2. Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(sentences)

# ----------------------------
# 3. Train classifier
# ----------------------------
model = LogisticRegression()
model.fit(X, labels)

# ----------------------------
# 4. Responses
# ----------------------------
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!"],
    "goodbye": ["Goodbye!", "See you later!"],
    "name": ["I'm your ML chatbot.", "You can call me AI Buddy."],
    "help": ["I can chat and answer simple questions."]
}

# ----------------------------
# 5. Memory system
# ----------------------------
memory = {
    "user_name": None
}

# ----------------------------
# 6. Chat loop
# ----------------------------
while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("🤖 Chatbot: Goodbye!")
        break

    # Simple name memory
    if "my name is" in user_input.lower():
        name = user_input.split("my name is")[-1].strip()
        memory["user_name"] = name
        print(f"🤖 Chatbot: Nice to meet you, {name}!")
        continue

    if "what is my name" in user_input.lower():
        if memory["user_name"]:
            print(f"🤖 Chatbot: Your name is {memory['user_name']}.")
        else:
            print("🤖 Chatbot: I don't know your name yet.")
        continue

    # ML prediction
    user_vector = vectorizer.transform([user_input])
    probabilities = model.predict_proba(user_vector)[0]
    prediction = model.predict(user_vector)[0]
    confidence = max(probabilities)

    # Confidence threshold
    if confidence < 0.4:
        print("🤖 Chatbot: I'm not confident about that. Could you rephrase?")
        continue

    print("🤖 Chatbot:", random.choice(responses.get(prediction, ["I'm not sure."])))

