import os
import tensorflow as tf

def load_model(base_dir, model_name='ResNet56.h5'):
    return tf.keras.models.load_model(os.path.join(base_dir, model_name))

def save_model(model, model_dir, model_name):
    path = os.path.join(model_dir, model_name)
    model.save(path)
    print(f"Model saved at {path}")
