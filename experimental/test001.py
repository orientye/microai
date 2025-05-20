import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 1. Get a function
            x, y = f.input, f.output  # 2. Get the function's input/output
            x.grad = f.backward(y.grad)  # 3. Call the function's backward

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Cube(Function):
    def forward(self, x):
        return x ** 3

    def backward(self, gy):
        x = self.input.data
        gx = 3 * (x ** 2) * gy
        return gx


class FourthPower(Function):
    def forward(self, x):
        return x ** 4

    def backward(self, gy):
        x = self.input.data
        gx = 4 * (x ** 3) * gy
        return gx


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def f(x):
    A = Square()
    B = Cube()
    C = FourthPower()
    return C(B(A(x)))


# forward and numerical_diff
x = Variable(1)
dy = numerical_diff(f, x)
print(dy)

# backward
x = Variable(1)
y = f(x)
y.backward()
print(x.grad)
