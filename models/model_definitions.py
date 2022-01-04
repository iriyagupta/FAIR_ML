import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


# Set up Baseline Model
def make_baseline_model(input_shape, primary_hidden_units, activation):
    model = tf.keras.Sequential([
      layers.Input(shape=(input_shape)),
      layers.Dense(primary_hidden_units[0], activation=activation),
      layers.Dense(primary_hidden_units[1], activation=activation),  
      layers.Dense(1, activation='sigmoid')
    ])
    return(model)

# Set up Primary Model
def make_primary_model(input_shape, primary_hidden_units, activation):
    model = tf.keras.Sequential([
      layers.Input(shape=(input_shape)),
      layers.Dense(primary_hidden_units[0], activation=activation),
      layers.Dense(primary_hidden_units[1], activation=activation),  
      layers.Dense(1, activation='sigmoid')
    ])
    return(model)

# Set up Adversarial Model
def make_adversarial_model(input_shape, adversary_hidden_units, activation):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_shape)),
        layers.Dense(adversary_hidden_units, activation=activation),
        layers.Dense(1, use_bias=True), # Linear layer
    ])
    return(model)