import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

# Import TensorFlow components
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore

# Download WordNet data if not already downloaded
nltk.download('wordnet', quiet=True)

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []

# Define list of characters to ignore
ignore_letters = ['?', '!', '.', ',']

# Loop through intents to extract patterns and tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words in each pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents as a tuple of words list and intent tag
        documents.append((word_list, intent['tag']))
        # Add intent tag to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignore_letters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words)) # Remove duplicates and sort

# Sort classes
classes = sorted(set(classes))

# Save words and classes to pickle files (optional)
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training set, bag of words for each pattern
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Convert training list to NumPy array
training = np.array(training)

# Separate input (X) and output (Y)
train_x = np.array([entry[0] for entry in training])
train_y = np.array([entry[1] for entry in training])

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))  # Output layer adjusted to len(classes)

# Compile the model with Stochastic Gradient Descent optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model to file
model.save('chatbot_model.h5')

print("Model created and saved")
