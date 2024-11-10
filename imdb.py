from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras import optimizers, losses  # Import optimizers and losses
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset, only the top 10,000 words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Reverse word index for debugging and visualizing the original text
word_index = imdb.get_word_index()
word_index = {k: (v+3) for k,v in word_index.items()}
reverse_word_index = dict([(value, key) for (key,value) in word_index.items()])
decoded_review = [reverse_word_index.get(i, '?') for i in train_data[0]]
print(decoded_review)

# Vectorize the sequences
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # Create an all-zero matrix
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # Set specific indices of results[i] to 1s
    return results

# Vectorize the train and test data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Convert labels to float32
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Build the model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with corrected optimizer and loss
model.compile(optimizer=optimizers.RMSprop(learning_rate=0.01),
              loss=losses.binary_crossentropy, 
              metrics=["acc"])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs=10,batch_size=512,validation_data=(x_val, y_val))
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)
plt.figure(1)
plt.plot(epochs, loss_values, 'ro', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
plt.show()
plt.figure(2)
acc= history.history['acc']
val_acc=history.history['val_acc']

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()


# # Summary of the model
# model.summary()

# model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=3, batch_size=64)

# scores = model.evaluate(test_data, test_labels, verbose=0)
# print(f"Accuracy: {scores[1] * 100:.2f}%")

# sample_review = test_data[0].reshape(1, -1)  # Example of using a test review
# prediction = model.predict(sample_review)
# print(f"Predicted sentiment: {'Positive' if prediction > 0.5 else 'Negative'}")