#!/usr/bin/env python3

import torch

N, n0, n1, n2 = 20, 10, 7, 13

# Some fake input
A0 = torch.randn(N, n0)
Y = torch.randn(N, n2)

# Layer 1
W1 = torch.randn(n1, n0, requires_grad=True)
b1 = torch.randn(n1, requires_grad=True)
temp1 = A0 @ W1.T
Z1 = temp1 + b1
A1 = torch.sigmoid(Z1)

# Layer 2
W2 = torch.randn(n2, n1, requires_grad=True)
b2 = torch.randn(n2, requires_grad=True)
temp2 = A1 @ W2.T
Z2 = temp2 + b2
A2 = torch.sigmoid(Z2)

# Loss and backward propagation
temp3 = A2 - Y
temp4 = temp3 ** 2
loss = temp4.mean()
loss.backward()