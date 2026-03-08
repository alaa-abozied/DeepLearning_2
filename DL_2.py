import numpy as np


def softmax(x):
    """Computes softmax values for a vector x."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


# 1. Initialize Weight Matrices

# Hidden-to-hidden weight matrix (3x3)
Wh = np.array([
    [0.1, 0.5, 0.1],
    [0.5, 0.9, 0.3],
    [0.3, 0.2, 0.1]
])

# Input-to-hidden weight matrix (3x4)
Wx = np.array([
    [0.6, 0.8, 0.4, 0.8],
    [0.2, 0.2, 0.8, 0.7],
    [0.9, 0.8, 0.1, 0.2]
])

# Hidden-to-output weight matrix (4x3)
Wy = np.array([
    [0.9, 0.8, 0.3],
    [0.2, 0.3, 0.4],
    [0.6, 0.9, 0.1],
    [0.5, 0.0, 0.3]
])


# 2. Define Initial State and Inputs

# Initial hidden state (3x1)
h0 = np.zeros((3, 1))

# Input vector (one-hot for "d")
x1 = np.array([[1], [0], [0], [0]])   # one-hot لـ 'd'
x2 = np.array([[0], [1], [0], [0]])   # one-hot لـ 'o'
x3 = np.array([[0], [0], [1], [0]])   # one-hot لـ 'g'

h = np.zeros((3, 1))
for t, x in enumerate([x1, x2, x3], 1):
    a = np.dot(Wh, h) + np.dot(Wx, x)
    h = np.tanh(a)
    y = softmax(np.dot(Wy, h))
    print(f"t={t}: y = {np.round(y.flatten(), 4)}")



# 3. Forward Propagation

# a1 = Wh*h0 + Wx*x1
a1 = np.dot(Wh, h0) + np.dot(Wx, x1)

# h1 = tanh(a1)
h1 = np.tanh(a1)

# y1 = softmax(Wy*h1)
y1 = softmax(np.dot(Wy, h1))


# Print results
print("Hidden Nodes (a1):")
print(a1)

print("\nHidden State (h1) [Rounded]:")
print(np.round(h1, 2))

print("\nPrediction Vector (y1) [Rounded]:")
print(np.round(y1, 2))