import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import tkinter


from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents(1).json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res



import tkinter
from tkinter import *
import tkinter as tk
from tkinter import Scrollbar, Text, Button, Canvas


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", tk.END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n', "user")
        ChatLog.config(state=tk.DISABLED)
    
        res = chatbot_response(msg)
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "Bot: " + res + '\n\n', "bot")
        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)
        
       
        
base = tk.Tk()
base.title("ChatBRO")
base.geometry("400x500")
base.resizable(width=True, height=True)

# Create Chat window
ChatLog = Text(
    base,
    bd=0,
    bg="#f5f5f5",  # Light gray background
    height="8",
    width="50",
    font=("Helvetica", 12),
)
ChatLog = Text(wrap="word", bd=1, insertborderwidth=2, highlightthickness=2)
ChatLog.config(state='normal', relief="flat")
ChatLog.tag_configure("user", foreground="#007BFF")  # Blue color for user
ChatLog.tag_configure("bot", foreground="#28A745")  # Green color for bot

ChatLog.config(state=tk.DISABLED)


scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog["yscrollcommand"] = scrollbar.set






 
SendButton = Button(
    base,
    font=("Helvetica", 12, 'bold'),
    text="Send",
    width="10",
    height=5,
    bd=0,
    bg="#4CAF50", 
    activebackground="#45A049",  
    fg="#ffffff",  
    command=send,
)

def on_click(event):
    if EntryBox.get("1.0", "end-1c") == "Type your message here":
        EntryBox.delete("1.0", END)
        EntryBox.config(fg='black')  # Change text color to black

def on_leave(event):
    if not EntryBox.get("1.0", "end-1c").strip():
        EntryBox.insert("1.0", "Type your message here")
        EntryBox.config(fg='grey')  # Change text color to grey

def send_message(event=None):
    message = EntryBox.get("1.0", "end-1c").strip()
    if message != '':
        ChatLog.config(state='normal')
        ChatLog.insert(END, "You: " + message + '\n\n')
        ChatLog.config(state='disabled')

        # Your processing logic for the message goes here

        EntryBox.delete("1.0", END)

# Your existing code for EntryBox
EntryBox = Text(
    base,
    bd=0,
    bg="#ffffff",  # White background
    width="29",
    height="5",
    font=("Helvetica", 12),
    borderwidth=2,  # Adjust the borderwidth
    highlightthickness=2,  # Adjust the highlightthickness
    relief="groove",
    
)

EntryBox.insert("1.0", "Type your message here")
EntryBox.config(fg='grey')  # Set text color to grey initially

# Bind events for placeholder-like behavior
EntryBox.bind("<FocusIn>", on_click)
EntryBox.bind("<FocusOut>", on_leave)

# Bind the 'Return' key to the send_message function
EntryBox.bind("<Return>", send_message)

scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=276, y=401, height=85)


def maximize_window(event):
    screen_width = base.winfo_screenwidth()
    screen_height = base.winfo_screenheight()
    scrollbar.place(x=screen_width - 324, y=6, height=screen_height - 192)  # Adjust values accordingly
    ChatLog.place(x=6, y=6, height=screen_height - 192, width=screen_width - 340)  # Adjust values accordingly
    EntryBox.place(x=6, y=screen_height - 181, height=90, width=screen_width - 20)  # Adjust values accordingly
    SendButton.place(x=screen_width - 260, y=screen_height - 181, height=85)  # Adjust values accordingly




base.bind('<Control-M>', maximize_window)
base.bind('<Control-m>', maximize_window)

base.mainloop()




