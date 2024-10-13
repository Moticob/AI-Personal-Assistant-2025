import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

class AIAssistant:
    def __init__(self, model_path, tokenizer_path, intents_path):
        self.model = load_model(model_path)
        with open(tokenizer_path, 'r') as file:
            self.tokenizer = tokenizer_from_json(json.load(file))
        with open(intents_path, 'r') as file:
            self.intents = json.load(file)

    def predict_intent(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20)
        predictions = self.model.predict(padded_sequences)
        intent_index = np.argmax(predictions)
        print(f"Predicted intent index: {intent_index}")  # Debugging line
        return intent_index

    def get_response(self, intent_index):
        intent_tag = self.intents['intents'][intent_index]['tag']
        print(f"Predicted intent tag: {intent_tag}")  # Debugging line
        for intent_data in self.intents['intents']:
            if intent_data['tag'] == intent_tag:
                return random.choice(intent_data['responses'])
        return "Sorry, I didn't understand that."

    def chat(self, text):
        intent_index = self.predict_intent(text)
        response = self.get_response(intent_index)
        return response

if __name__ == "__main__":
    assistant = AIAssistant('models/assistant_model.h5', 'models/tokenizer.pickle', 'data/intents.json')
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        print("Assistant:", assistant.chat(user_input))

