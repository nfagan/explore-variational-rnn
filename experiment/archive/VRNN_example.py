import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import os
import helper_VRNN_example as helper
from math import sqrt, log10
import random 
import json
import pandas as pd
import sys

# Parse command line arguments
lambda_string = sys.argv[1]
lambda_values = [float(x) for x in lambda_string.split(',')]

alpha_string = sys.argv[2]
alpha_values = [float(x) for x in alpha_string.split(',')]

beta_string = sys.argv[3]
beta_values = [float(x) for x in beta_string.split(',')]

dir_name = sys.argv[4]
epochs = int(sys.argv[5])
input_type = sys.argv[6]

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Model hyperparameters
latent_dim = 60     # Dimension of the latent space in the variational autoencoder
output_dim = 128    # Dimension of the decoder output (matches rnn_units)
reward_output_dim = 30  # Dimension for reward estimatinos
rnn_units = 128     # Number of units in the RNN layers
time_steps = 30     # Number of time steps in the sequence
input_dim = 1       # Dimension of each input at each time step
N = time_steps      # Number of nodes
feature_dim = 1     # Dimension of features for input data

reconstruction_mse_scaling_factor = 50
mse_loss_scaling_factor = 0.0001
action_loss_scaling_factor = 0.01
kl_loss_scaling_factor = 0.0001



# Training parameters
trials_per_epoch = 120
batch_size = 300

# Decision tree dictionary defining the structure of possible paths
decision_tree = {
    '0': {'right': [-1, '1'], 'left': [3, '7'],'up': [-1, '13'], 'down': [2, '19'], 'up1': [-1, '25']},
    '1': {'up': [-1, '2'], 'down': [2, '3'], 'right': [-1, '4'], 'left': [3, '5'],'up1': [-1, '6']},
    '2': {},
    '3': {},
    '4': {},
    '5': {},
    '6': {},
    '7': {'up': [-1, '8'], 'down': [2, '9'], 'right': [-1, '10'], 'left': [3, '11'],'up1': [-1, '12']},
    '8': {},
    '9':{},
    '10':{},
    '11': {},
    '12':{},
    '13': {'up': [-1, '14'], 'down': [2, '15'], 'right': [-1, '16'], 'left': [3, '17'],'up1': [-1, '18']},
    '14': {},
    '15': {},
    '16': {},
    '17': {},
    '18': {},
    '19': {'up': [-1, '20'], 'down': [2, '21'], 'right': [-1, '22'], 'left': [3, '23'],'up1': [-1, '24']},
    '20': {},
    '21': {},
    '22': {},
    '23': {},
    '24': {},
    '25': {'up': [-1, '26'], 'down': [2, '27'], 'right': [-1, '28'], 'left': [3, '29'],'up1': [-1, '30']},
    '26': {},
    '27': {},
    '28': {},
    '29': {},
    '30': {}
}

# Generate all path analysis data
results = helper.analyze_tree_paths(decision_tree)
path_names, path_leaf_dict, sibling_map, node_path_map, node_path_name, path_indices, node_indices, est_best_path_map, path_node_map = results

index_path_map = {path_indices[i]: node_indices[i] for i in range(len(path_indices))}
num_path = len(path_names) # Number of paths

def build_encoder(input_dim, latent_dim):
    """
    Build the encoder network that outputs parameters for the latent distribution and a sampled latent vector.
    
    Args:
        input_dim: Dimension of the input features
        latent_dim: Dimension of the latent space
        
    Returns:
        A Keras model that takes inputs and outputs z_mean, z_log_var, and sampled z
    """
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    model = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return model

def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    
    Args:
        args: Tuple containing z_mean and z_log_var tensors
        
    Returns:
        A sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_decoder(latent_dim, output_dim):
    """
    Build the decoder network that maps the latent vector back to the observable space.
    
    Args:
        latent_dim: Dimension of the latent space
        output_dim: Dimension of the output
        
    Returns:
        A Keras model that takes latent vectors and outputs reconstructions
    """
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(latent_inputs)
    outputs = layers.Dense(output_dim, activation='linear')(x)
    model = models.Model(latent_inputs, outputs, name='decoder')
    return model


class VariationalRNN(tf.keras.Model):
    """
    A Variational Recurrent Neural Network (VRNN) model that combines
    variational autoencoders with recurrent neural networks.
    """
    
    def __init__(self, encoder, decoder, rnn_units, time_steps=N, num_path=num_path, alpha=0.0, beta=1.0, lambda_=1.0):
        """
        Initialize the VRNN model.
        
        Args:
            encoder: Encoder model
            decoder: Decoder model
            rnn_units: Number of RNN units
            time_steps: Number of time steps
            num_path: Number of paths in the decision tree
            alpha: Weight for MSE loss
            beta: Weight for action loss
            lambda_: Weight for KL loss
        """
        super(VariationalRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dense = tf.keras.layers.Dense(time_steps, activation="linear", 
                                         kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.dense_action = tf.keras.layers.Dense(num_path, activation='softmax', 
                                                kernel_initializer='glorot_uniform')
        self.prior_mean_layer = tf.keras.layers.Dense(latent_dim, activation=None)
        self.prior_log_var_layer = tf.keras.layers.Dense(latent_dim, activation=None)
        self.rnn_units = rnn_units
        self.time_steps = time_steps
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_

    def compute_prior(self, hidden_state):
        """
        Compute the prior mean and variance given the hidden state,
        with value clamping to prevent numerical instability.
        
        Args:
            hidden_state: Current hidden state
            
        Returns:
            Tuple of (prior_mean, prior_var)
        """
        prior_mean = self.prior_mean_layer(hidden_state)
        prior_log_var = self.prior_log_var_layer(hidden_state)

        # Clamp values to reasonable ranges
        prior_mean = tf.clip_by_value(prior_mean, -10.0, 10.0)
        prior_log_var = tf.clip_by_value(prior_log_var, -10.0, 10.0)
        prior_var = tf.exp(prior_log_var)

        return prior_mean, prior_var
    
    def call(self, inputs, training=True):
        """
        Forward pass through the VRNN model.
        
        Args:
            inputs: Input tensor of shape [batch_size, time_steps, feature_size]
            training: Whether in training mode
            
        Returns:
            Tuple of (outputs, total_loss, first_decoder_loss, second_decoder_loss, action_output, kl_d)
        """
        batch_size = tf.shape(inputs)[0]
        # Model outputs across all timeline
        all_h = []
        all_z_means = []
        all_z_log_vars = []
        all_reconstruction_outputs = []
        action_outputs = []

        prior_values = tf.constant([0] * N, dtype=tf.float32)
        prior_values = tf.tile(tf.reshape(prior_values, [1, -1]), [batch_size, 1])
        
        if len(prior_values.shape) == 2:
            reshaped_prior_values = tf.reshape(prior_values, [batch_size, N, 1])
        else:
            reshaped_prior_values = prior_values
            
        output = reshaped_prior_values
        total_loss = 0
        first_decoder_loss = 0
        second_decoder_loss = 0
        z = tf.zeros([batch_size, latent_dim], dtype=tf.float32)
        kl_d = 0
        
        for t in range(self.time_steps):
            # Extract current timestep data
            x_t = inputs[:, t, :]
            one_hot_t = tf.one_hot(t, self.time_steps)
            one_hot_t = tf.tile(tf.reshape(one_hot_t, [1, -1]), [batch_size, 1])

            # Initialize hidden state for first timestep
            if t == 0:
                hidden_state = tf.zeros((batch_size, self.rnn_units))
                
            hidden_state_flat = tf.reshape(hidden_state, [batch_size, -1])
            z = tf.reshape(z, [batch_size, -1])

            # Prepare inputs for encoder
            encoder_input = tf.concat([x_t, one_hot_t, hidden_state_flat], axis=1)
            z_mean, z_log_var, z = self.encoder(encoder_input)

            all_z_means.append(z_mean)
            all_z_log_vars.append(z_log_var)

            # Decode latent state
            decoder_input = tf.concat([z], axis=1)
            hidden_state_flat = self.decoder(decoder_input)
            hidden_state_flat = tf.clip_by_value(hidden_state_flat, -10.0, 10.0)
            
            hidden_state = tf.reshape(hidden_state_flat, [batch_size, 1, self.rnn_units])

            # Generate reconstruction
            rec_decoder_input = tf.concat([hidden_state_flat], axis=1)
            rec_output = self.dense(rec_decoder_input)
            rec_output = tf.clip_by_value(rec_output, -4.5, 4.5)
            
            if len(rec_output.shape) == 2:
                rec_output = tf.reshape(rec_output, [batch_size, N, 1])

            # Combine predicted and prior values
            rec_slice = rec_output[:, :t+1, :]
            prior_slice = reshaped_prior_values[:, t+1:, :]
            rec_output = tf.concat([rec_slice, prior_slice], axis=1)
            all_reconstruction_outputs.append(rec_output)

            # Generate action output
            action_decoder_input = tf.concat([hidden_state_flat], axis=1)
            action_output = self.dense_action(action_decoder_input)
            
            # Calculate KL divergence with prior
            prior_mean, prior_var = self.compute_prior(hidden_state_flat)
            kl_loss = self.calculate_kl_loss(z_mean, z_log_var, prior_mean, prior_var)
            kl_d += kl_loss

        # Add losses during training
        if training:
            all_z_means = tf.stack(all_z_means, axis=1)
            all_z_log_vars = tf.stack(all_z_log_vars, axis=1)
            
            # Scale and add KL loss
            kl_loss = kl_d * self.lambda_ * kl_loss_scaling_factor
            self.add_loss(kl_loss)

            # Add action loss
            action_loss = -helper.calculate_V(inputs, action_output, N, num_path, index_path_map, path_map, self) * self.beta * action_loss_scaling_factor
            self.add_loss(action_loss)
            
            # Add MSE loss
            mse_loss = self.compute_mse_loss(inputs, all_reconstruction_outputs, t) * mse_loss_scaling_factor
            self.add_loss(self.alpha * mse_loss)
            
            # Calculate total losses
            total_loss += sum(self.losses)
            first_decoder_loss += kl_loss + action_loss + (self.alpha * mse_loss)
            second_decoder_loss += reconstruction_mse_scaling_factor * mse_loss

        outputs = tf.stack(all_reconstruction_outputs, axis=1)
        return outputs, total_loss, first_decoder_loss, second_decoder_loss, action_output, kl_d

    def calculate_kl_loss(self, z_means, z_log_vars, prior_mean, prior_var, epsilon=1e-6):
        """
        Calculate KL divergence between the posterior and prior distributions.
        
        Args:
            z_means: Mean of the posterior distribution
            z_log_vars: Log variance of the posterior distribution
            prior_mean: Mean of the prior distribution
            prior_var: Variance of the prior distribution
            epsilon: Small value for numerical stability
            
        Returns:
            KL divergence loss
        """
        # Clamp values for numerical stability
        z_log_vars = tf.clip_by_value(z_log_vars, -5.0, 5.0)
        prior_var = tf.clip_by_value(prior_var, 1e-6, 1e2)

        prior_log_var = tf.math.log(prior_var + epsilon)
        z_var = tf.exp(z_log_vars) + epsilon

        kl_loss = -0.5 * tf.reduce_sum(
            1.0 + z_log_vars - prior_log_var
            - (tf.square(z_means - prior_mean) + z_var) / (prior_var + epsilon),
            axis=1
        )
        return tf.reduce_mean(kl_loss)

    def compute_mse_loss(self, inputs, outputs, t):
        """
        Compute the MSE loss between inputs and outputs.
        
        Args:
            inputs: Input tensor
            outputs: Output tensor
            t: Current timestep
            
        Returns:
            MSE loss value
        """
        inputs = tf.convert_to_tensor(inputs)
        outputs = tf.convert_to_tensor(outputs)
        
        # Reshape inputs to match outputs shape
        inputs_expanded = tf.expand_dims(inputs, axis=1)
        inputs_expanded = tf.tile(inputs_expanded, [1, time_steps, 1, 1])
        inputs_expanded = tf.transpose(inputs_expanded, perm=[1, 0, 2, 3])

        # Calculate absolute difference for current timestep
        squared_diff = tf.abs(inputs_expanded[t] - outputs[t])
        
        return tf.reduce_sum(squared_diff)

def train_step(model, optimizer, clip_value=0.5):
    """
    Perform one training step with the model.
    
    Args:
        model: The VRNN model
        optimizer: The optimizer to use
        clip_value: Maximum gradient value for clipping
        
    Returns:
        Total loss value
    """
    # Separate parameters for different loss components
    first_decoder_params = model.decoder.trainable_variables + model.encoder.trainable_variables + \
                          model.prior_mean_layer.trainable_variables + model.prior_log_var_layer.trainable_variables + \
                          model.dense_action.trainable_variables
    second_decoder_params = model.dense.trainable_variables

    # Create a persistent gradient tape
    with tf.GradientTape(persistent=True) as tape:
        # Generate input data based on specified type
        if input_type == "uniform":
            values = np.array([[random.uniform(-4.5, 4.5) for i in range(time_steps)] for _ in range(batch_size)])
        elif input_type == "normal":
            values = np.array([helper.get_truncated_normal_samples(size=time_steps, mean=0, sd=sqrt(5), 
                                                                  low=-4.5, upp=4.5) for _ in range(batch_size)])
        elif input_type == "binary":
            values = np.array([[random.choice([-4, 4]) for i in range(time_steps)] for _ in range(batch_size)])
        else:
            values = np.array([helper.get_truncated_normal_samples(size=time_steps, mean=0, sd=sqrt(5), 
                                                                  low=-4.5, upp=4.5) for _ in range(batch_size)])
            
        # Prepare input tensor
        input_data = tf.constant(values, dtype=tf.float32)
        input_data = tf.reshape(input_data, [batch_size, time_steps, feature_dim])

        # Forward pass through the model
        reconstructed, total_loss, first_decoder_loss, second_decoder_loss, _, _ = model(input_data, training=True)

    # Compute gradients
    first_decoder_gradients = tape.gradient(first_decoder_loss, first_decoder_params)
    second_decoder_gradients = tape.gradient(second_decoder_loss, second_decoder_params)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # Apply gradient clipping
    clipped_first_decoder_gradients = [tf.clip_by_value(g, -clip_value, clip_value) 
                                      if g is not None else None for g in first_decoder_gradients]
    clipped_second_decoder_gradients = [tf.clip_by_value(g, -clip_value, clip_value) 
                                       if g is not None else None for g in second_decoder_gradients]
    
    # Apply gradients to the parameters
    optimizer.apply_gradients(zip(clipped_first_decoder_gradients, first_decoder_params))
    optimizer.apply_gradients(zip(clipped_second_decoder_gradients, second_decoder_params))
    
    # Clean up the persistent tape
    del tape

    return total_loss

def train_model(model, epochs, trials_per_epoch):
    """
    Train the model for a specified number of epochs and trials.
    
    Args:
        model: The VRNN model
        epochs: Number of epochs to train
        trials_per_epoch: Number of trials per epoch
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(trials_per_epoch):
            loss = train_step(model, optimizer)
            epoch_loss += loss

# Prepare path mapping and covariance matrix
path_map = np.zeros((num_path, N), dtype=int)
for i in range(num_path):
    path_map[i, :] = [1 if f"{n+1}" in path_names[i] else 0 for n in range(N)]
path_map = tf.convert_to_tensor(path_map, dtype=tf.float32)


# Main training loop for all hyperparameter combinations
for beta in beta_values:
    for lambda_ in lambda_values:
        for alpha in alpha_values:
            print("lambda: " + str(lambda_) + "alpha: " + str(alpha))
            model_name = 'seq_vrnn_model_tpa_lambda:' + str(lambda_) + "_alpha:" + str(alpha) + "_beta:" + str(beta) + "_example"

            # Initialize model components
            encoder = build_encoder(time_steps + input_dim + rnn_units, latent_dim)
            decoder = build_decoder(latent_dim, rnn_units)

            # Create and compile the VRNN model
            vrnn_model = VariationalRNN(encoder, decoder, rnn_units, time_steps=time_steps, 
                                      alpha=alpha, beta=beta, lambda_=lambda_)
            vrnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

            # Train the model
            train_model(vrnn_model, epochs, trials_per_epoch)
            
            # Save model weights
            if os.path.exists(dir_name + model_name + '_weights.tf'):
                os.remove(dir_name + model_name + '_weights.tf')
            vrnn_model.save_weights(dir_name + model_name + '_weights.tf')

            tf.print("model saved to: ", dir_name + model_name + '_weights.tf')