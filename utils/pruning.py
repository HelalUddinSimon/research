import numpy as np
import tensorflow as tf
from utils.training import train_step

def threshold_prune_weights(prune_model, dominant_eigenvectors, percentile, x_train, y_train, x_test, y_test,
                            fine_tune_epochs=100, learning_rate=0.001, batch_size=32):
    vector_index = 0
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    masks = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for layer in prune_model.layers:
        if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
            for weight_tensor in layer.trainable_weights:
                weights_array = weight_tensor.numpy()
                flat_weights = weights_array.flatten()
                flat_eigenvector = dominant_eigenvectors[vector_index].flatten()
                vector_index += 1

                num_weights = len(flat_weights)
                num_significant_weights = int(num_weights * (100 - percentile) / 100)
                num_less_significant_weights = num_weights - num_significant_weights

                sorted_indices = np.argsort(np.abs(flat_eigenvector))
                less_significant_indices = sorted_indices[:num_less_significant_weights]
                significant_indices = sorted_indices[num_significant_weights:]

                mask = np.ones_like(flat_weights, dtype=bool)
                mask[less_significant_indices] = False

                if len(significant_indices) > 0:
                    for index in less_significant_indices:
                        corresponding_index = significant_indices[index % len(significant_indices)]
                        flat_weights[corresponding_index] += flat_weights[index] * 0.5

                flat_weights[less_significant_indices] = 0
                pruned_weights_array = flat_weights.reshape(weights_array.shape)
                mask_array = mask.reshape(weights_array.shape)
                weight_tensor.assign(pruned_weights_array)
                masks.append(mask_array)
        else:
            masks.append(None)

    for epoch in range(fine_tune_epochs):
        print(f"Fine-tuning epoch {epoch + 1}/{fine_tune_epochs}")
        for step, (x_batch, y_batch) in enumerate(train_data):
            val_batch = next(iter(val_data))
            train_step(prune_model, x_batch, y_batch, loss_fn, optimizer, val_batch)

    return masks
