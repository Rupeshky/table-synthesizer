import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

class Sampling(Layer):
    """Sampling z in the latent space."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAEModel(Model):
    """Custom model where the VAE loss is added."""
    def __init__(self, original_dim, intermediate_dim, latent_dim, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Sequential([
            Dense(intermediate_dim, activation='relu'),
            Dense(latent_dim * 2),
        ])
        self.decoder = Sequential([
            Dense(intermediate_dim, activation='relu'),
            Dense(original_dim, activation='sigmoid')
        ])
        self.sampling = Sampling()

    def call(self, inputs):
        z_mean, z_log_var = tf.split(self.encoder(inputs), num_or_size_splits=2, axis=1)
        z = self.sampling((z_mean, z_log_var))
        reconstructed = self.decoder(z)
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(inputs, reconstructed), axis=-1))
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)
        return reconstructed

# Load and preprocess data
df = pd.read_csv('adult.csv')

# Define column transformers
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features),
    ])

df_processed = preprocessor.fit_transform(df)

# Define VAE model parameters
original_dim = df_processed.shape[1]
intermediate_dim = 64
latent_dim = 2

# Initialize and compile the VAE model
vae = VAEModel(original_dim, intermediate_dim, latent_dim)
vae.compile(optimizer='adam')

# Train the VAE
vae.fit(df_processed.toarray(), df_processed.toarray(), epochs=100, batch_size=256, verbose=2)

# Generate synthetic data
n_samples = 1000
z_sample = np.random.normal(size=(n_samples, latent_dim))
generated_data = vae.decoder(z_sample).numpy()

# Split the generated data back into numerical and categorical parts
num_data = generated_data[:, :len(numeric_features)]
cat_data = generated_data[:, len(numeric_features):]

# Inverse transform for numerical data
num_data_inv = preprocessor.named_transformers_['num'].inverse_transform(num_data)

# Initialize an empty list to collect the inverse transformed categorical data
cat_data_inv = []

# Iterate through the transformers to inverse transform the categorical data
for transformer in preprocessor.transformers_:
    # Unpack the transformer tuple
    name, transformer_instance, columns = transformer
    if name == 'cat':  # Check if it's the categorical transformer
        # Inverse transform the categorical data
        # Assuming the output of the transformer is at the end of the cat_data array
        cat_data_transformed = transformer_instance.inverse_transform(cat_data)
        cat_data_inv.append(cat_data_transformed)

# Concatenate all the inverse transformed data
cat_data_inv = np.concatenate(cat_data_inv, axis=1)
num_data_inv=np.rint(num_data_inv)
inverse_transformed_data = np.hstack([num_data_inv, cat_data_inv])

# Create a DataFrame for the inversely transformed data
column_names = numeric_features + categorical_features  # Ensure correct order and naming
synthetic_df = pd.DataFrame(inverse_transformed_data, columns=column_names)

# Handling numerical data rounding if necessary
for col in numeric_features:
    # Ensure the column is in a numerical format
    if pd.api.types.is_numeric_dtype(synthetic_df[col]):
        synthetic_df[col] = synthetic_df[col].astype(float).round().astype(int)
    else:
        print(f"Column {col} is not numeric. Skipping rounding.")

# Save the synthetic data
synthetic_df.to_csv('synthetic_data.csv', index=False)
print("Synthetic data saved to synthetic_data.csv")



