import os
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam

from utils.data_loader import load_cifar10
from utils.model_utils import load_model, save_model
from utils.flops_estimator import estimate_layer_flops
from utils.weight_counter import count_zero_non_zero_weights
from utils.hessian import power_iteration_for_hessian_vector_product
from utils.pruning import threshold_prune_weights
from utils.training import lr_schedule
from utils.plot_utils import plot_flops_comparison, plot_metrics_comparison, plot_eigenvector_distribution

if __name__ == '__main__':
    percentile = 50
    model_dir = 'saved_models'
    base_dir = 'base_model'
    os.makedirs(model_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    model = load_model(base_dir)

    initial_loss, initial_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Initial Accuracy: {initial_accuracy * 100:.2f}%")

    layer_names = [layer.name for layer in model.layers]
    layer_flops = [estimate_layer_flops(layer) for layer in model.layers]
    print("Total FLOPS before pruning:", sum(layer_flops))

    zeros, non_zeros = count_zero_non_zero_weights(model)
    print("Before Pruning - Zeros:", zeros, "Non-Zeros:", non_zeros)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    dominant_eigenvector = power_iteration_for_hessian_vector_product(model, dataset)

    masks = threshold_prune_weights(model, dominant_eigenvector, percentile, x_train, y_train, x_test, y_test)

    pruned_loss, pruned_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy after pruning: {pruned_accuracy * 100:.2f}%")

    zeros_after, non_zeros_after = count_zero_non_zero_weights(model)
    print("After Pruning - Zeros:", zeros_after, "Non-Zeros:", non_zeros_after)

    pruned_layer_flops = [estimate_layer_flops(layer, is_pruned=True) for layer in model.layers]
    print("Total FLOPS after pruning:", sum(pruned_layer_flops))

    fig1 = plot_flops_comparison(layer_flops, pruned_layer_flops, layer_names)
    fig1.savefig(os.path.join(model_dir, 'flops_comparison.png'))

    save_model(model, model_dir, f'pruned_model_{percentile}.h5')

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[LearningRateScheduler(lr_schedule)])

    final_loss, final_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Accuracy after fine-tuning: {final_accuracy * 100:.2f}%")

    save_model(model, model_dir, f'finetune_model_{percentile}.h5')

    fig2 = plot_metrics_comparison(['Base', 'Pruned', 'Finetuned'], [initial_loss, pruned_loss, final_loss], [initial_accuracy, pruned_accuracy, final_accuracy])
    fig2.savefig(os.path.join(model_dir, 'metrics_comparison.png'))

    fig3 = plot_eigenvector_distribution(dominant_eigenvector)
    fig3.savefig(os.path.join(model_dir, 'eigenvector_distribution.png'))
