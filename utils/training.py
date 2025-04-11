import tensorflow as tf

def train_step(model, x_batch, y_batch, loss_fn, optimizer, val_data=None):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_batch, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    val_accuracy = None
    if val_data is not None:
        x_val_batch, y_val_batch = val_data
        val_predictions = model(x_val_batch, training=False)
        val_correct_predictions = tf.equal(tf.argmax(val_predictions, 1), tf.argmax(y_val_batch, 1))
        val_accuracy = tf.reduce_mean(tf.cast(val_correct_predictions, tf.float32))

    return loss, accuracy, val_accuracy

def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    elif epoch < 20:
        return 0.0001
    else:
        return 0.00001
