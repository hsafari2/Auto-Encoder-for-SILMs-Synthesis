import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2

# Load and preprocess data
rawData = pd.read_excel('/work/bavarian/hsafari2/CO2Absorption/Processed_Sorption_Data.xlsx')
rawData['Mol. Wt.'] = pd.to_numeric(rawData['Mol. Wt.'], errors='coerce')
rawData['moles of CO2/kg sorbent'] = rawData['moles of CO2/kg IL'] / 2
rawData = rawData.drop(columns=['moles of CO2/kg IL'])
rawData = pd.get_dummies(rawData, columns=['Host Polymer'])
X = rawData[['Temp. (°C)', 'Pressure (psi)', 'Mol. Wt.'] + [col for col in rawData.columns if col.startswith('Host Polymer_')]]
y = rawData['moles of CO2/kg sorbent']

# Scale the features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Define the CVAE model
class CVAE(keras.Model):
    def __init__(self, input_dim, condition_dim, latent_dim, l1_reg=0.001, l2_reg=0.001, dropout_rate=0.2):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.encoder = self.get_encoder(input_dim, condition_dim, latent_dim)
        self.decoder = self.get_decoder(latent_dim, condition_dim, input_dim)

    def get_encoder(self, input_dim, condition_dim, latent_dim):
        inputs = keras.Input(shape=(input_dim,))
        conditions = keras.Input(shape=(condition_dim,))
        x = layers.Concatenate()([inputs, conditions])
        x = layers.Dense(64, activation="relu", kernel_regularizer=l1(self.l1_reg), activity_regularizer=l2(self.l2_reg))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation="relu", kernel_regularizer=l1(self.l1_reg), activity_regularizer=l2(self.l2_reg))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        return keras.Model([inputs, conditions], [z_mean, z_log_var, z], name="encoder")

    def get_decoder(self, latent_dim, condition_dim, output_dim):
        latent_inputs = keras.Input(shape=(latent_dim,))
        conditions = keras.Input(shape=(condition_dim,))
        x = layers.Concatenate()([latent_inputs, conditions])
        x = layers.Dense(32, activation="relu", kernel_regularizer=l1(self.l1_reg), activity_regularizer=l2(self.l2_reg))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=l1(self.l1_reg), activity_regularizer=l2(self.l2_reg))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(output_dim, bias_initializer=keras.initializers.Constant(0.1))(x)
        return keras.Model([latent_inputs, conditions], outputs, name="decoder")

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        y, conditions = inputs
        z_mean, z_log_var, z = self.encoder([y, conditions])
        reconstructed = self.decoder([z, conditions])
        kl_loss = -0.5 * keras.backend.sum(1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var), axis=-1)
        self.add_loss(keras.backend.mean(kl_loss))
        return reconstructed

# Define the loss function
def cvae_loss(y_true, y_pred):
    mse_loss = keras.losses.mse(y_true, y_pred)
    return keras.backend.mean(mse_loss)

# Create and compile the model
input_dim = 1  # CO2 adsorption
condition_dim = X_scaled.shape[1]  # Number of input parameters
latent_dim = 2
l1_reg = 0.001
l2_reg = 0.001
dropout_rate = 0.2
cvae = CVAE(input_dim, condition_dim, latent_dim, l1_reg, l2_reg, dropout_rate)
cvae.compile(optimizer='adam', loss=cvae_loss)

# Train the CVAE
early_stopping = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True, verbose=0)
history = cvae.fit(
    [y_scaled.reshape(-1, 1), X_scaled], 
    y_scaled, 
    epochs=1500,
    batch_size=32, 
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0  # Set this to 0 to hide epoch-by-epoch output
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Training completed. Loss plot generated.")



def generate_candidate_pool(n_samples=1000, extend_factor=1.5):
    new_inputs = []
    host_polymer_cols = [col for col in X.columns if col.startswith('Host Polymer_')]
    
    temp_min, temp_max = X['Temp. (°C)'].min(), X['Temp. (°C)'].max()
    pressure_min, pressure_max = X['Pressure (psi)'].min(), X['Pressure (psi)'].max()
    mol_wt_min, mol_wt_max = X['Mol. Wt.'].min(), X['Mol. Wt.'].max()
    
    temp_range = temp_max - temp_min
    pressure_range = pressure_max - pressure_min
    mol_wt_range = mol_wt_max - mol_wt_min
    
    for _ in range(n_samples):
        temp = np.random.uniform(temp_min - temp_range * (extend_factor - 1) / 2, 
                                 temp_max + temp_range * (extend_factor - 1) / 2)
        pressure = np.random.uniform(pressure_min - pressure_range * (extend_factor - 1) / 2, 
                                     pressure_max + pressure_range * (extend_factor - 1) / 2)
        mol_wt = np.random.uniform(mol_wt_min - mol_wt_range * (extend_factor - 1) / 2, 
                                   mol_wt_max + mol_wt_range * (extend_factor - 1) / 2)
        
        # Ensure non-negative values
        temp = max(0, temp)
        pressure = max(0, pressure)
        mol_wt = max(0, mol_wt)
        
        host_polymer = np.zeros(len(host_polymer_cols))
        host_polymer[np.random.randint(0, len(host_polymer))] = 1
        new_inputs.append(np.concatenate(([temp, pressure, mol_wt], host_polymer)))
    
    return pd.DataFrame(new_inputs, columns=['Temp. (°C)', 'Pressure (psi)', 'Mol. Wt.'] + host_polymer_cols)





def predict_adsorption(cvae, conditions, batch_size=32, n_samples=10):
    predictions = []
    for _ in range(n_samples):
        z = tf.random.normal(shape=(conditions.shape[0], cvae.latent_dim))
        batch_predictions = cvae.decoder.predict([z, conditions])
        predictions.append(batch_predictions)
    predictions = np.mean(np.array(predictions), axis=0)
    return scaler_y.inverse_transform(predictions)

def estimate_uncertainty(cvae, conditions, batch_size=32, n_samples=10):
    predictions = []
    for _ in range(n_samples):
        z = tf.random.normal(shape=(conditions.shape[0], cvae.latent_dim))
        batch_predictions = cvae.decoder.predict([z, conditions])
        predictions.append(batch_predictions)
    return np.std(np.array(predictions), axis=0)

def select_promising_candidates(candidates, n_select=10, uncertainty_weight=0.2):
    candidates['Score'] = candidates['Predicted CO2 Adsorption'] + uncertainty_weight * candidates['Prediction Uncertainty']
    return candidates.sort_values('Score', ascending=False).head(n_select)

def get_host_polymer(row):
    host_polymer_cols = [col for col in row.index if col.startswith('Host Polymer_')]
    return host_polymer_cols[row[host_polymer_cols].argmax()].replace('Host Polymer_', '')

# Generate, predict, and select promising candidates
candidate_pool = generate_candidate_pool()
candidate_pool_scaled = scaler_X.transform(candidate_pool)
predicted_adsorption = predict_adsorption(cvae, candidate_pool_scaled)
candidate_pool['Predicted CO2 Adsorption'] = predicted_adsorption
prediction_uncertainty = estimate_uncertainty(cvae, candidate_pool_scaled)
candidate_pool['Prediction Uncertainty'] = scaler_y.inverse_transform(prediction_uncertainty)
promising_candidates = select_promising_candidates(candidate_pool)
promising_candidates['Host Polymer'] = promising_candidates.apply(get_host_polymer, axis=1)

# Print the most promising candidates
print("Top 10 Most Promising SILM Candidates:")
print(promising_candidates[['Temp. (°C)', 'Pressure (psi)', 'Mol. Wt.', 'Host Polymer', 'Predicted CO2 Adsorption', 'Prediction Uncertainty']])

# Optional: Save the promising candidates to a CSV file
#promising_candidates.to_csv('promising_silm_candidates.csv', index=False)

print("\nThe most promising SILM candidates have been identified and saved to 'promising_silm_candidates.csv'.")
print("These candidates are recommended for future experimental work to validate their CO2 adsorption performance.")
print("Remember to iterate this process by incorporating new experimental results back into the model training data.")


print("\n--- Top 10 Actual SILMs with Highest CO2 Adsorption ---")

# Sort the original data by CO2 adsorption in descending order
top_actual_silms = rawData.sort_values('moles of CO2/kg sorbent', ascending=False).head(10)

# Function to get the Host Polymer name from one-hot encoded columns for the original data
def get_host_polymer_original(row):
    host_polymer_cols = [col for col in row.index if col.startswith('Host Polymer_')]
    return host_polymer_cols[row[host_polymer_cols].argmax()].replace('Host Polymer_', '')

# Add Host Polymer name to the top actual SILMs
top_actual_silms['Host Polymer'] = top_actual_silms.apply(get_host_polymer_original, axis=1)

# Select and reorder columns for display
columns_to_display = ['Temp. (°C)', 'Pressure (psi)', 'Mol. Wt.', 'Host Polymer', 'moles of CO2/kg sorbent']
top_actual_silms_display = top_actual_silms[columns_to_display]

# Rename the CO2 adsorption column to match the format used for artificial data
top_actual_silms_display = top_actual_silms_display.rename(columns={'moles of CO2/kg sorbent': 'CO2 Adsorption'})

# Display the top 10 actual SILMs
print(top_actual_silms_display.to_string(index=False))

# Optionally, save these top actual SILMs to a CSV file
#top_actual_silms_display.to_csv('top_actual_silms.csv', index=False)

print("\nThe top 10 actual SILMs with highest CO2 adsorption have been identified and saved to 'top_actual_silms.csv'.")



