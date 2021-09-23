#!/usr/bin/env python
# coding: utf-8

# # Chatbot for E-commerce
# 
# ## 1. Data Analysis

# In[98]:


import json
import nltk
from nltk.stem import WordNetLemmatizer
import random
import numpy as np

words=[]
classes = []
documents = []

ignore_words = ['?', '!', ".", ",", "'s"]

data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:
        #add documents in the corpus
        words.append(pattern)
        documents.append((pattern, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[101]:


print("Classes: ", classes, "\n")
print("Words: ", words, "\n")
print("Documents: ", documents, "\n")


# lemmatizer = WordNetLemmatizer()
#
#
# # In[103]:
#
#
# # lemmatize, lower each word and remove duplicates
words = [w.lower() for w in words if w not in ignore_words]
words = sorted(list(set(words)))

print(words)
#
#
# # In[104]:
#
#
# # sort classes
classes = sorted(list(set(classes)))

#
#
# # In[105]:
#
#
# # words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
#
#
# # In[106]:
#
#
with open("words_data.json", "w", encoding='utf-8') as words_data:
    json.dump(words, words_data)
print(classes)

# with open("classes_data.json", "w") as classes_data:
#     json.dump(classes, classes_data)
#
#
# # ## 3. Create training and testing data
#
# # In[108]:
#
#
# # create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [word.lower() for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# # In[110]:
#
#
for t in training:
    print(t)

random.shuffle(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - intents
X_train = list(training[:,0])
y_train = list(training[:,1])
print("Training data created")

for i in range(len(X_train)):
    print(X_train[i], y_train[i])
#
#
# # ## 4 Build the Sequential Model
#
# # In[115]:
#
#
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# # equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

#
# # In[117]:
#
#
# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")
#
#
# # In[120]:
#
#
# # plot_model(model, to_file = "model.png", show_shapes=True)
#
#
# # In[ ]:
#
#
#
#
#
# # ## Testing
#
# # In[95]:
#
#
# def clean(sentence):
#     # tokenize the pattern - split words into array
#     sentence_words = nltk.word_tokenize(sentence)
#     # stem each word - create short form for word
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words
#
# # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
# def bow(sentence, words):
#     # tokenize the pattern
#     sentence_words = clean(sentence)
#     # bag of words - matrix of N words, vocabulary matrix
#     bag = [0]*len(words)
#     for s in sentence_words:
#         for i,w in enumerate(words):
#             if w == s:
#                 # assign 1 if current word is in the vocabulary position
#                 bag[i] = 1
#     return(np.array(bag))
#
# def predict_class(sentence, model):
#     # filter out predictions below a threshold
#     p = bow(sentence, words)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.5
# #     print(res)
#
#     results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list
#
#
# # In[96]:
#
#
# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if(i['tag']== tag):
#             result = random.choice(i['responses'])
#             break
#     return result
# def chatbot_response(text):
#     ints = predict_class(text, model)
#     predict_intent = ints[0]['intent']
#     # print(ints)
#     res = getResponse(ints, intents)
#
#     return res, ints
#
#
# # In[97]:
#
#
# while(True):
#     w = str(input())
#     print("You: ", w)
#     res = chatbot_response(w)
#     print("Bot: ",res)
    



