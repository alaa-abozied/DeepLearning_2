

# 🧠 Recurrent Neural Network (RNN) From Scratch using NumPy

## 📌 Project Overview

This project demonstrates how a **Recurrent Neural Network (RNN)** works internally by implementing the **forward propagation process from scratch using NumPy only**.

The goal is to understand the mathematical operations inside RNNs without relying on deep learning frameworks such as TensorFlow or PyTorch.

The model processes a small sequence of one-hot encoded inputs representing characters and produces prediction probabilities using **Softmax**.

This implementation focuses on:

- Understanding hidden state transitions
- Matrix multiplication in RNNs
- Activation functions
- Sequential data processing
- Softmax probability outputs

---

# 🧠 RNN Architecture

The network consists of three main weight matrices:

### 1️⃣ Input → Hidden
Wx ∈ R^(3×4)



### 2️⃣ Hidden → Hidden (Recurrent Connection)
Wh ∈ R^(3×3)



### 3️⃣ Hidden → Output
Wy ∈ R^(4×3)



At each timestep the network computes:
a_t = Wh * h_(t-1) + Wx * x_t



Hidden state:
h_t = tanh(a_t)



Prediction:
y_t = softmax(Wy * h_t)



---

# 🔤 Input Representation

The sequence used in this example:
d → o → g



Each character is represented using **one-hot encoding**.

Example:
d = [1,0,0,0]
o = [0,1,0,0]
g = [0,0,1,0]



---

# ⚙️ Implementation Details

- Language: **Python**
- Library: **NumPy**
- Model Type: **Vanilla RNN**
- Activation Function: **tanh**
- Output Function: **Softmax**
- Sequence Length: **3 timesteps**

---

# 🧮 Mathematical Flow

For each timestep:

1️⃣ Combine previous hidden state with current input
a = Wh * h + Wx * x



2️⃣ Apply activation function
h = tanh(a)



3️⃣ Generate prediction
y = softmax(Wy * h)



---
