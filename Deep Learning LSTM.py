import numpy as np

def softmax_func(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e, axis=0)

Wh = np.array([
    [0.4, 0.7, 0.2],
    [0.6, 0.1, 0.5],
    [0.3, 0.8, 0.4]
])

Wx = np.array([
    [0.5, 0.6, 0.7, 0.3],
    [0.4, 0.9, 0.2, 0.8],
    [0.7, 0.5, 0.3, 0.6]
])

Wy = np.array([
    [0.8, 0.4, 0.2],
    [0.3, 0.7, 0.5],
    [0.6, 0.1, 0.9],
    [0.2, 0.5, 0.4]
])

h_prev = np.array([
    [0],
    [0],
    [0]
])

x_input = np.array([
    [1],
    [0],
    [0],
    [0]
])

a1 = np.dot(Wh, h_prev) + np.dot(Wx, x_input)
print("a_1 (Hidden Nodes):\n", a1)
print("-" * 30)

h1 = np.tanh(a1)
print("h_1 (New Hidden State):\n", np.round(h1, 2))
print("-" * 30)

y_raw = np.dot(Wy, h1)
y_pred = softmax_func(y_raw)

print("y_1 (Predicted Probabilities):\n", np.round(y_pred, 2))
