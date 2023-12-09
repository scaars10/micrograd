import math


def tanh(value):
    return (math.e ** (2 * value) - 1) / (math.e ** (2 * value) + 1)


def sigmoid(value):
    return 1 / (1 + math.e ** (-value))

