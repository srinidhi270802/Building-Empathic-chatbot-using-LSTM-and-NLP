import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model
import random
import openai  # Import OpenAI module


completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are some famous astronomical observatories?"}
  ]
)

# Set up your OpenAI GPT-3 API key
openai.api_key = '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'

# Load your existing chatbot model
model = load_model('chatbot_model.h5')

# Load the words and classes used in your chatbot
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load your intents data
data_file = open('intents(1).json').read()
intents = json.loads(data_file)

# Extract patterns and responses to create the 'documents' list
documents = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        documents.append((pattern, intent['tag'], intent['responses']))

# Set up lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    # Tokenize and lemmatize the user's input
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Return the bag of words: 0 or 1 for each word in the bag that exists in the sentence
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return(np.array(bag))

def predict_class(sentence):
    # Filter out predictions below a certain threshold (e.g., 0.25)
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_gpt_response(user_input):
    # Call ChatGPT to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=150
    )
    return response['choices'][0]['text']

# Example conversation loop
print("Chatbot: Hello! Ask me something or type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        break

    # Use your chatbot model to predict intent
    predictions = predict_class(user_input)

    # Choose the most probable intent
    intent = predictions[0]['intent']

    # If intent is identified, respond using your chatbot model
    if intent in classes:
        bot_response = "I'm sorry, I don't understand."
        for doc in documents:
            if doc[1] == intent:
                bot_response = random.choice(doc[2])
        print(f"Chatbot: {bot_response}")
    else:
        # If intent is not identified, use ChatGPT for a generic response
        gpt_response = get_gpt_response(user_input)
        print(f"Chatbot: {gpt_response}")
