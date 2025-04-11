import numpy as np
import tensorflow as tf

def estimate_layer_flops(layer, is_pruned=False):
    if isinstance(layer, tf.keras.layers.Conv2D):
        output_shape = layer.output_shape[1:3]
        kernel_size = layer.kernel_size
        input_channels = layer.input_shape[-1]
        output_channels = layer.output_shape[-1]

        flops_per_instance = 2 * np.prod(kernel_size) * input_channels * output_channels
        total_flops = flops_per_instance * np.prod(output_shape)

        if is_pruned:
            sparsity = (layer.weights[0].numpy() == 0).mean()
            total_flops *= (1 - sparsity)

        return total_flops

    elif isinstance(layer, tf.keras.layers.Dense):
        input_size = layer.input_shape[-1]
        output_size = layer.output_shape[-1]
        total_flops = 2 * input_size * output_size

        if is_pruned:
            sparsity = (layer.weights[0].numpy() == 0).mean()
            total_flops *= (1 - sparsity)

        return total_flops

    return 0
