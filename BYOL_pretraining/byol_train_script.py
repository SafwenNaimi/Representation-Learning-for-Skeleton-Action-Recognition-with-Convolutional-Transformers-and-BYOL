import tensorflow as tf
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense, LSTM, TimeDistributed, AveragePooling1D
from transformer import TransformerEncoder, PatchClassEmbedding, Patches
from data import load_data, load_kinetics, random_flip, random_noise, one_hot
    
X_train, y_train, X_test, y_test = load_data('vitpose', 1, 
                                                    legacy=False, verbose=False)
print(X_train.shape)
        
def byol_train_model(online_network, target_network, X_train, optimizer, batch_size, transformation_function, epochs=100, verbose=0):
    epoch_wise_loss = []

    for epoch in range(epochs):
        step_wise_loss = []

        # Make a batched dataset
        batched_data = get_batched_data(X_train, batch_size)
        for data_batch in batched_data:

            # Apply transformation
            transform_1 = transformation_function(data_batch)
            transform_2 = transformation_function(data_batch)

            with tf.GradientTape(persistent=True) as tape:
                # Forward propagation through online and target networks
                projection_1 = online_network(transform_1, training=True)
                projection_2 = target_network(transform_2, training=True)
                #with tf.device('cpu'):
                #target_projection_1 = target_network(transform_1, training=True)
                #target_projection_2 = target_network(transform_2, training=True)

                # Compute BYOL loss (cosine similarity)
                loss = byolloss(projection_1, projection_2)

            # Compute gradients and update online network
            gradients = tape.gradient(loss, online_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, online_network.trainable_variables))

            step_wise_loss.append(loss.numpy())

        epoch_wise_loss.append(np.mean(step_wise_loss))

        if verbose > 0:
            print("Epoch: {} Loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

        # Update target network with exponential moving average
        update_target_network(online_network, target_network, momentum=0.9)

    return online_network, epoch_wise_loss

def update_target_network(online_network, target_network, momentum):
    for online_weight, target_weight in zip(online_network.weights, target_network.weights):
        target_weight.assign(momentum * target_weight + (1.0 - momentum) * online_weight)

d_models= 192
#n_patches= 30
def create_base_model(input_shape, model_name="base_model"):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape, name='input')
    print("this is the input")
    print(inputs.shape)
    x = tf.keras.layers.Conv1D(filters=192, kernel_size=1, activation='selu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = tf.keras.layers.Conv1D(filters=192, kernel_size=1, activation='selu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = tf.keras.layers.Dense(192)(x)
    x = PatchClassEmbedding(d_model= d_models)(x)
    transformer = TransformerEncoder(d_model= d_models, num_heads= 3, d_ff=768, dropout=0.3, activation = tf.nn.gelu, n_layers= 6)
    x = transformer(x)
    x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
    x = tf.keras.layers.Dense(256)(x)
    
    return tf.keras.Model(inputs, x, name=model_name)


def byolloss(p, z):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)


def ceiling_division(n, d):
    """
    Ceiling integer division
    """
    return -(n // -d)

def get_batched_dataset_generator(data, batch_size):

    num_bathes = ceiling_division(data.shape[0], batch_size)
    for i in range(num_bathes):
        yield data[i * batch_size : (i + 1) * batch_size]

    # return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))

def get_batched_data(X, batch_size):
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X[excerpt]




def scaling_transform_vectorized(X, sigma=0.3):
    """
    Scaling by a random factor
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=X.shape)
    return X * scaling_factor


def noise_transform_vectorized(X, sigma=0.05):
    """
    Adding random Gaussian noise with mean 0
    """
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def flip_transform(X, horizontal=True, vertical=False):
    if horizontal and np.random.choice([True, False]):
        X = np.flip(X, axis=1)
    if vertical and np.random.choice([True, False]):
        X = np.flip(X, axis=0)
    return X


def generate_composite_transform_function_simple(transform_funcs):

    for i, func in enumerate(transform_funcs):
        print(i, func)
    def combined_transform_func(sample):
        for func in transform_funcs:
            sample = func(sample)
        return sample
    return combined_transform_func


batch_size = 64
decay_steps = 1000
epochs = 100
temperature = 0.1
transform_funcs = [
    noise_transform_vectorized,
    #scaling_transform_vectorized,
    flip_transform, # Use Scaling trasnformation (BEST) 
]
transformation_function = generate_composite_transform_function_simple(transform_funcs)


def attach_byol_head(base_model, hidden_1=256, hidden_2=128, hidden_3=64): #_3:64
    """
    Attach a 3-layer fully-connected encoding head

    Architecture:
        base_model
        -> Dense: hidden_1 units
        -> ReLU
        -> Dense: hidden_2 units
        -> ReLU
        -> Dense: hidden_3 units
    """

    input = base_model.input
    x = base_model.output

    projection_1 = tf.keras.layers.Dense(hidden_1)(x)
    projection_1 = tf.keras.layers.Activation("relu")(projection_1)
    projection_2 = tf.keras.layers.Dense(hidden_2)(projection_1)
    projection_2 = tf.keras.layers.Activation("relu")(projection_2)
    projection_3 = tf.keras.layers.Dense(hidden_3)(projection_2)

    byol_model = tf.keras.Model(input, projection_3, name= base_model.name + "_byol")

    return byol_model

input_shape = (25, 60) ##52 for reduced keypoints

# Create the base model
base_model = create_base_model(input_shape)


start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.01, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
# transformation_function = byol_utitlities.generate_combined_transform_function(trasnform_funcs_vectorized, indices=trasnformation_indices)

base_model = create_base_model(input_shape, model_name="base_model")
byol_model = attach_byol_head(base_model)
byol_model.summary()

trained_byol_model, epoch_losses = byol_train_model(byol_model,byol_model, X_train, optimizer, batch_size, transformation_function, epochs=epochs, verbose=1)

byol_model_save_path = "BYOL_pretraining/model.hdf5"
trained_byol_model.save(byol_model_save_path) 
