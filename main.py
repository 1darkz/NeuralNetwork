import numpy as np


#neural network structure
input_size = 2
hidden_size = 2
output_size = 1

#weights and biases
np.random.seed(42)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.random.randn(hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.random.randn(output_size)

#activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#derivation of activation function
def sigmoid_derivative(x):
    return x * (1 - x)

#mean squared error loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#forward pass
def forward_pass(x):
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    return hidden_layer_output, output

#training function with backward pass and weight updates
def train(X, y, learning_rate, epochs):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

    for epoch in range(epochs):
        #forward pass
        hidden_output, predictions = forward_pass(X)

        #calculate loss
        loss = mse_loss(y, predictions)

        #backward pass
        error = y - predictions
        d_output = error * sigmoid_derivative(predictions)

        error_hidden_layer = d_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

        #update weights and biases
        weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
        bias_output += np.sum(d_output, axis=0) * learning_rate
        weights_input_hidden += X.T.dot(d_hidden_layer) + learning_rate
        bias_hidden += np.sum(d_hidden_layer, axis=0) * learning_rate

        #print loss at every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

#Example data
#input data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

#output labels
y = np.array([[0], [1], [1], [0]])

#train the model
train(X, y, learning_rate=0.1, epochs=10000)

#test the model
def predict(X):
    _, output = forward_pass(X)
    return output

#test on training data
predictions = predict(X)
print("Predictions:")
print(predictions)

#convert output to binary
binary_predictions = (predictions > 0.5).astype(int)
print("Binary Predictions:")
print (binary_predictions)