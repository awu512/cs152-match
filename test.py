from __future__ import annotations

import unittest

import match
from match import Matrix

# Use torch to compute correct output
import torch
from torch import Tensor


def almostEqual(matrix: Matrix, tensor: Tensor, check_grad=False) -> bool:
    m = to_tensor(matrix, get_grad=check_grad)
    t = Tensor(tensor.grad) if check_grad else tensor
    return torch.allclose(m, t, rtol=1e-02, atol=1e-05)


def to_tensor(matrix: Matrix, requires_grad=False, get_grad=False) -> Tensor:
    mdata = matrix.grad.data if get_grad else matrix.data.data
    return torch.tensor(mdata, requires_grad=requires_grad)


def mat_ten(dim1, dim2) -> tuple[Matrix, Tensor]:
    mat = match.randn(dim1, dim2)
    ten = to_tensor(mat, requires_grad=True)
    return mat, ten


def neuron(a, w, b, relu=True):
    z = a @ w.T + b.T
    a = z.relu() if relu else z.sigmoid()
    return z, a


class TestMatch(unittest.TestCase):
    def test_3layer(self):
        """Test the output and gradient of a three layer network."""

        N = 5
        n0 = 4
        n1 = 3
        n2 = 6
        n3 = 1

        # Fake input and output
        x = mat_ten(N, n0)
        y = mat_ten(N, 1)

        # Parameters
        W = []
        b = []

        # Layer 1
        W.append(mat_ten(n1, n0))
        b.append(mat_ten(n1, 1))

        # Layer 2
        W.append(mat_ten(n2, n1))
        b.append(mat_ten(n2, 1))

        # Layer 3
        W.append(mat_ten(n3, n2))
        b.append(mat_ten(n3, 1))

        # Forward
        mat_a, ten_a = x
        for i, ((mat_W, ten_W), (mat_b, ten_b)) in enumerate(zip(W, b)):
            mat_z, mat_a = neuron(mat_a, mat_W, mat_b, relu=(i < len(W) - 1))
            ten_z, ten_a = neuron(ten_a, ten_W, ten_b, relu=(i < len(W) - 1))
            self.assertTrue(almostEqual(mat_z, ten_z))
            self.assertTrue(almostEqual(mat_a, ten_a))

        # MSE Loss
        mat_y, ten_y = y
        mat_loss = ((mat_a - mat_y) ** 2).mean()
        ten_loss = ((ten_a - ten_y) ** 2).mean()
        self.assertTrue(almostEqual(mat_loss, ten_loss))

        # Backward
        mat_loss.backward()
        ten_loss.backward()

        # Check all gradients (even input and output)
        grads = [y] + W + b + [x]
        for mat_g, ten_g in grads:
            self.assertTrue(almostEqual(mat_g, ten_g, check_grad=True))

    def test_arithmetic(self):
        """Test the output and gradient of arbitrary arithmetic."""

        mat1, ten1 = mat_ten(3, 2)
        mat2, ten2 = mat_ten(3, 2)

        mat3 = mat1 * mat2 * -1 + 5
        ten3 = ten1 * ten2 * -1 + 5
        self.assertTrue(almostEqual(mat3, ten3))

        mat4 = mat3.sigmoid()
        ten4 = ten3.sigmoid()
        self.assertTrue(almostEqual(mat4, ten4))

        mat5 = (mat4 / mat1) ** 3
        ten5 = (ten4 / ten1) ** 3
        self.assertTrue(almostEqual(mat5, ten5))

        mat6 = mat5.sigmoid()
        ten6 = ten5.sigmoid()
        self.assertTrue(almostEqual(mat6, ten6))

        mat7 = mat6.sum()
        ten7 = ten6.sum()
        self.assertTrue(almostEqual(mat7, ten7))

        mat7.backward()
        ten7.backward()
        self.assertTrue(almostEqual(mat1, ten1, check_grad=True))
        self.assertTrue(almostEqual(mat2, ten2, check_grad=True))

    def test_nn(self):
        """Test the neural network layer objects."""
        N, n0, n1 = 7, 10, 14

        mat_linr = match.nn.Linear(n0, n1)
        mat_relu = match.nn.ReLU()

        ten_linr = torch.nn.Linear(n0, n1)
        ten_relu = torch.nn.ReLU()

        # Manually set the tensor to the same values as the matrix
        ten_linr.weight = torch.nn.Parameter(to_tensor(mat_linr.A))
        ten_linr.bias = torch.nn.Parameter(to_tensor(mat_linr.b).squeeze())

        mat_x, ten_x = mat_ten(5, 10)

        mat_z = mat_linr(mat_x)
        mat_a = mat_relu(mat_z)

        ten_z = ten_linr(ten_x)
        ten_a = ten_relu(ten_z)

        self.assertTrue(almostEqual(mat_z, ten_z))
        self.assertTrue(almostEqual(mat_a, ten_a))


if __name__ == "__main__":
    unittest.main()