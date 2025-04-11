import tensorflow as tf

def initialize_vectors_correctly(model):
    vectors = []
    for layer in model.layers:
        if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
            for weight in layer.trainable_weights:
                total_params = tf.size(weight).numpy()
                vector = tf.random.normal(shape=(total_params,))
                vectors.append(tf.reshape(vector, weight.shape))
    return vectors

def power_iteration_for_hessian_vector_product(model, dataset, num_iterations=10):
    for inputs, targets in dataset:
        v = initialize_vectors_correctly(model)
        for _ in range(num_iterations):
            with tf.GradientTape() as outer_tape:
                for vector in v:
                    outer_tape.watch(vector)
                with tf.GradientTape() as inner_tape:
                    predictions = model(inputs, training=True)
                    loss = tf.keras.losses.categorical_crossentropy(targets, predictions)
                grads = inner_tape.gradient(loss, model.trainable_weights)
                grad_v_product = tf.add_n([tf.reduce_sum(g * vector) for g, vector in zip(grads, v)])
            hessian_v_product = outer_tape.gradient(grad_v_product, v)
            v = [hv / (tf.norm(hv) + 1e-10) for hv in hessian_v_product]
        return [vec.numpy().flatten() for vec in v]
