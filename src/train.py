import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load intents
with open('data/intents.json') as file:
    intents = json.load(file)

# Prepare training data
texts = []
labels = []
label_map = {}
for intent in intents['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)
        labels.append(intent['tag'])
    if intent['tag'] not in label_map:
        label_map[intent['tag']] = len(label_map)

# Tokenize texts
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=20)

# Convert labels to one-hot encoding
label_sequences = np.array([label_map[label] for label in labels])
one_hot_labels = tf.keras.utils.to_categorical(label_sequences, num_classes=len(label_map))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, one_hot_labels, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, input_shape=(20,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=300, batch_size=8, validation_data=(X_val, y_val))

# Save model and tokenizer
model.save('models/assistant_model.h5')
with open('models/tokenizer.pickle', 'w') as file:
    json.dump(tokenizer.to_json(), file)

