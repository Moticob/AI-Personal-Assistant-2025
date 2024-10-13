import unittest
from src.assistant import AIAssistant

class TestAIAssistant(unittest.TestCase):
    def setUp(self):
        self.assistant = AIAssistant('models/assistant_model.h5', 'models/tokenizer.pickle', 'data/intents.json')

    def test_greeting(self):
        response = self.assistant.chat("Hi")
        self.assertIn(response, ["Hello!", "Hi there!", "Greetings!", "How can I assist you today?"])

if __name__ == '__main__':
    unittest.main()

