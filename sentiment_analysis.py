import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data preparation (you'll need to replace this with your actual dataset)
texts = ["I love this! 😊", "This is bad 😡", "Amazing experience 🤩", "Not good 😞"]
sentiment_labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative (binary classification)
regression_targets = [0.9, 0.2, 0.8, 0.3]  # Example regression targets

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

# Convert data to numpy arrays
X = np.array(padded_sequences)
y_class = np.array(sentiment_labels)
y_regress = np.array(regression_targets)

# Define model architecture
input_layer = Input(shape=(10,), name='input')

# Embedding layer
embedding = Embedding(input_dim=10000, output_dim=64, input_length=10)(input_layer)

# LSTM layer for shared representation
shared_lstm = LSTM(64, return_sequences=False)(embedding)

# Dropout for regularization
dropout = Dropout(0.5)(shared_lstm)

# Classification output
class_output = Dense(1, activation='sigmoid', name='class_output')(dropout)

# Regression output
regression_output = Dense(1, name='regression_output')(dropout)

# Define the model
model = Model(inputs=input_layer, outputs=[class_output, regression_output])

# Compile the model with different losses for each task
model.compile(optimizer='adam',
              loss={'class_output': 'binary_crossentropy', 'regression_output': 'mean_squared_error'},
              metrics={'class_output': 'accuracy', 'regression_output': 'mse'})

# Print the model summary
model.summary()

# Train the model
history = model.fit(X, {'class_output': y_class, 'regression_output': y_regress},
                    epochs=10, batch_size=2, validation_split=0.2)

# Evaluate the model
loss, class_loss, reg_loss, class_acc, reg_mse = model.evaluate(X, {'class_output': y_class, 'regression_output': y_regress})
print(f'Loss: {loss}, Classification Accuracy: {class_acc}, Regression MSE: {reg_mse}')