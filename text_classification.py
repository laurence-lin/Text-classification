import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.imdb as imdb
from tensorflow import keras



'''
test classification: 
Movie review dataset


'''

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000) # select 10000 most frequent words in the training data
# dictionary that map words to integer index
word_index = imdb.get_word_index()  # word index for imdb dataset

print('Train entry:', train_data.shape, 'Train labels', train_labels.shape)
print('Test entry:', test_data.shape, 'Test labels', test_labels.shape)
# Each train sample is a movie review, and could be different length

word_index = {k:(v + 3) for k, v in word_index.items()} # create dictionary of word and index, but add index by 3
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

print('Number of words:', len(word_index.keys()))

# Get the reversed mapping: map integer index to word
reversed_index = {value:key for key, value in word_index.items()}


def decode_review(text):
    '''
    Decode the word index back to string word
    text: a list of word index number
    return: word string
    '''
    return ' '.join([reversed_index.get(i, '?') for i in text])

'''
Before input to network, convert the index integer to tensor
Do padding & truncating to make each review has same length
'''

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value = word_index['<PAD>'], # pad value with 0 if length too short
                                                        padding = 'post', # padding after the sequence
                                                        maxlen = 256) # set each lenght of sequence equal 256

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value = word_index['<PAD>'], # pad value with 0 if length too short
                                                       padding = 'post', # padding after the sequence
                                                       maxlen = 256)
    
print('length of each word index sequences: ', len(train_data[0]))

# Build the model: Use embedding layer
vocab_size = 10000
    
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16)) # Embedding words to word vector
model.add(keras.layers.GlobalAveragePooling1D()) # convert [batch, step, features] to [batch, features]
model.add(keras.layers.Dense(16, activation = tf.nn.relu))
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid)) # output use sigmoid, btw 0 and 1, to represent classes

print(model.summary())

# Model configuration
model.compile(optimizer = 'adam', 
              loss = 'binary_crossentropy',
              metrics = ['acc'])

# Create validate set to select the best performance model
x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Train the model
history = model.fit(x_train, 
          y_train,
          batch_size = 512,
          epochs = 40,
          validation_data = (x_val, y_val),
          verbose = 1
          )   

results = model.evaluate(test_data, test_labels) 

print(results) # return [error, accuracy]

# Show training curve 
train_history = history.history
print(train_history.keys())

acc = train_history['acc']
val_acc = train_history['val_acc']
loss = train_history['loss']
val_loss = train_history['val_loss']

fig = plt.figure(1)
plt.plot(acc, 'bo', label = 'Train accuracy')
plt.plot(val_acc, 'b-', label = 'Validate accuracy')
plt.xlabel('Train epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validate Accuracy')
plt.legend()

plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    







