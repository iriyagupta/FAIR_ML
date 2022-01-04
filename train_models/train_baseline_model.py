import time
import tensorflow as tf
import numpy as np

# Get primary model's loss function
def get_primary_loss(labels, classifier_logits, adversarial_weights):
    """
    Params:
      labels: The target labels
      classifier_logits: Primary model output
      adversarial_weights: Adversary model output
    Returns:
      primary loss
    """

    # Get the loss of the primary model
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=classifier_logits)
    
    # Add in the output of the adversarial model to compute full loss
    primary_weighted_loss = (adversarial_weights * loss)
    
    primary_weighted_loss = tf.reduce_mean(primary_weighted_loss)
    return primary_weighted_loss


def train(baseline_model, X_train, y_train, train_steps, primary_optimizer, batch_size):
    """
    Params:
      baseline_model
      X_train: Input for primary model
      y_train: Labels
      train_steps: How many times to run through all data
      primary_optimizer
      batch_size
    Returns:
      baseline model
    """

    # Run through one step of the baseline model
    def train_step(X_train_batch, y_train_batch):
        with tf.GradientTape() as gen_tape:
            classifier_logits = baseline_model(X_train_batch)

            # Just use vector of 1s for adversary weights if we are not using adversary model
            primary_loss = get_primary_loss(y_train_batch, classifier_logits, tf.ones_like(y_train_batch))

        primary_gradients = gen_tape.gradient(primary_loss, baseline_model.trainable_variables)
        primary_optimizer.apply_gradients(zip(primary_gradients, baseline_model.trainable_variables))
        
    # Split datasets into batches
    X_train_batches = np.array_split(X_train, batch_size, axis=0)
    y_train_batches = np.array_split(y_train, batch_size, axis=0)
    
    start = time.time()
    print("Training baseline model...")
    for step in range(train_steps):
        # Show progress
        if step % 10 == 0:
            print("Training step: ", step)
            
            # Note: we do not include eval data results here because we are saving all the test data for accurate test results
            # since there is limited data after preprocessing
            acc = tf.keras.metrics.Accuracy()
            class_predictions = tf.cast(tf.greater(baseline_model(X_train), 0.5), tf.float32)
            acc.update_state(class_predictions, y_train)
            print("Training accuracy: ", acc.result().numpy())

        for i in range(len(X_train_batches)):
            train_step(X_train_batches[i], y_train_batches[i])
    print("Done. Train time in seconds: ", time.time()-start)
    return baseline_model
