import numpy as np

def count_zero_non_zero_weights(model):
    total_zeros = 0
    total_non_zeros = 0
    for weight in model.weights:
        non_zeros = np.count_nonzero(weight.numpy())
        total_non_zeros += non_zeros
        zeros = np.size(weight.numpy()) - non_zeros
        total_zeros += zeros
    return total_zeros, total_non_zeros
