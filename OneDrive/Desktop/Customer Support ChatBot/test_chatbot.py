
import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random



model = tf.keras.models.load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints):
    if len(ints) == 0:
        return "I'm not sure what you mean. Could you please rephrase that?"
    
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

test_phrases = [
    "Hello there",
    "How can I return my order?",
    "What products do you have?",
    "Where is my order?",
    "Thanks for your help",
    "Goodbye"
]


print("Testing the chatbot with sample phrases:")
for phrase in test_phrases:
    ints = predict_class(phrase)
    response = get_response(ints)
    print(f"User: {phrase}")
    print(f"Bot: {response}")
    print("-" * 50)