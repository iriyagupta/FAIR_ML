import time

import tensorflow as tf
import time
from train_models import train_baseline_model
import numpy as np


# Adversary's loss function
def get_adversary_loss(labels, classifier_logits, adversarial_weights):
    """
    Params:
      labels: The target labels
      classifier_logits: Primary model output
      adversarial_weights: Adversary model output
    Returns:
      adversary loss
    """

    # Get the loss of the adversarial model based on the output of the primary classifier
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=classifier_logits)
    
    # Hinge loss: Another loss function the paper mentioned experimenting with for the adversary
    # but ultimately did not use. We also experimented with it and did not see much
    # change in results.
    # hinge_loss = tf.keras.losses.Hinge(reduction=tf.keras.losses.Reduction.NONE)
    # loss = tf.maximum(hinge_loss(labels, classifier_logits), 0.1) # Avoiding errors at 0

    # Update the loss with the adversary's weights
    # Since we are trying to maximize the adversary's loss, we multiply by -1
    adversary_weighted_loss = -(adversarial_weights * loss)
    return tf.reduce_mean(adversary_weighted_loss)
    return adversary_weighted_loss


        
def train(primary_model, 
          adversarial_model, 
          X_train, 
          X_train_adversarial, 
          y_train, 
          pretrain_steps, 
          train_steps, 
          primary_optimizer,
          adversary_optimizer,
          batch_size):
    
    """
    Params:
      primary_model
      adversarial_model
      X_train: Input for primary model
      X_train_adversarial: Input for adversarial model
      y_train: Labels
      pretrain_steps: How many steps to train primary model before training both primary and adversary together
      train_steps: How many times to run through all data
      primary_optimizer
      adversary_optimizer
      batch_size
    Returns:
      primary model, adversary model
    """

    
    # Run through one training step of the ARL model
    def train_step(X_train_batch, X_train_adversarial_batch, y_train_batch, primary_only=False):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            classifier_logits = primary_model(X_train_batch)
            if not primary_only:
                adversarial_weights = adversarial_model(X_train_adversarial_batch)
                # Rescale and center adversary output
                mean_adversarial_weights = tf.reduce_mean(adversarial_weights)
                adversarial_weights /= tf.maximum(mean_adversarial_weights, 1e-4)
                # From paper: "Add one to ensure that all training examples contribute to the loss"
                adversarial_weights = tf.ones_like(adversarial_weights)+adversarial_weights
                
            # Calculate losses
            if primary_only:
                # Just use vector of 1s for adversary weights if we are not using adversary model yet
                primary_loss = train_baseline_model.get_primary_loss(y_train_batch, classifier_logits, tf.ones_like(y_train_batch))
            else:
                primary_loss = train_baseline_model.get_primary_loss(y_train_batch, classifier_logits, adversarial_weights)
                adversary_loss = get_adversary_loss(y_train_batch, classifier_logits, adversarial_weights)
        primary_gradients = gen_tape.gradient(primary_loss, primary_model.trainable_variables)
        if not primary_only:
            adversary_gradients = disc_tape.gradient(adversary_loss, adversarial_model.trainable_variables)
        primary_optimizer.apply_gradients(zip(primary_gradients, primary_model.trainable_variables))
        if not primary_only:
            adversary_optimizer.apply_gradients(zip(adversary_gradients, adversarial_model.trainable_variables))
    
    # Split datasets into batches
    X_train_batches = np.array_split(X_train, batch_size, axis=0)
    X_train_adversarial_batches = np.array_split(X_train_adversarial, batch_size, axis=0)
    y_train_batches = np.array_split(y_train, batch_size, axis=0)

    start = time.time()
    
    # Pretrain primary model
    print("Pre-training primary model...")
    for step in range(pretrain_steps):
        # Show progress
        if step%10 == 0:
            print("Training step: ", step)
            # Note: we do not include eval data results here because we are saving all the test data for accurate test results
            # since there is limited data after preprocessing
            acc = tf.keras.metrics.Accuracy()
            class_predictions = tf.cast(tf.greater(primary_model(X_train), 0.5), tf.float32)
            acc.update_state(class_predictions, y_train)
            print("Training accuracy: ", acc.result().numpy())

        for i in range(len(X_train_batches)):
            train_step(X_train_batches[i], X_train_adversarial_batches[i], y_train_batches[i], primary_only=True)

    # Train primary and adversarially model
    print("Training adversarially...")
    for step in range(train_steps):        
        # Show progress
        if step%5 == 0:
            print("Training step: ", step)
            acc = tf.keras.metrics.Accuracy()
            class_predictions = tf.cast(tf.greater(primary_model(X_train), 0.5), tf.float32)
            acc.update_state(class_predictions, y_train)
            print("Training accuracy: ", acc.result().numpy())
        for i in range(len(X_train_batches)):
            train_step(X_train_batches[i], X_train_adversarial_batches[i], y_train_batches[i], primary_only=False)   

    print("Done. Train time in seconds: ", time.time()-start)
    return primary_model, adversarial_model
