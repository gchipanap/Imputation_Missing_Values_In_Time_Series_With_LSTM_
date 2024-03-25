import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from custom_loss import  Ltotal


# Función para generar el Residual Sum Vector (RSV)
def generate_RSV(model, X):
    hiddens = model.predict(X)
    max_length = max(item.shape[0] for item in hiddens)
    
    rsv = []
    for hidden in hiddens:
        pad_width = ((0, max_length - hidden.shape[0]), (0, 0))
        padded_hidden = np.pad(hidden, pad_width, mode='constant')
        rsv.append(padded_hidden)
    
    return np.concatenate(rsv, axis=1)


# Función para calcular zt+1
def calculate_zt_1(W_imprt, rsv_t):
    # Ajustar la forma de W_imprt si es necesario
    if len(W_imprt.shape) == 3:
        W_imprt = tf.squeeze(W_imprt, axis=[0, -1])

    return tf.matmul(W_imprt, rsv_t)


# Definir la arquitectura del modelo LSTM
def build_model(input_shape, output_shape, lstm_units, learning_rate=0.001, l2_penalty=0.001):
    inputs = Input(shape=input_shape)
    lstm_layer = LSTM(units=lstm_units, activation='sigmoid', return_sequences=True)(inputs)
    outputs = Dense(output_shape, kernel_regularizer=regularizers.l2(l2_penalty))(lstm_layer)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model


data = pd.read_csv('Data/energydata_complete.csv')

X = data[['T1']].values
y = data[['Appliances']].values

#Normalizar los datos
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
y_normalized = scaler.fit_transform(y)


# Dividir los datos en entrenamiento y prueba
split_index = int(len(X) * 0.8)
X_train, X_test = X_normalized[:split_index], X_normalized[split_index:]
y_train, y_test = y_normalized[:split_index], y_normalized[split_index:]

learning_rate = 0.01
l2_penalty = 0.01
batch_size = 256
epochs = 150

input_shape = (X_train.shape[1], 1)
inputs = Input(shape=(1, 1))
lstm_layer = LSTM(units=50, activation='sigmoid', return_sequences=True)(inputs)
outputs = Dense(1, kernel_regularizer=regularizers.l2(l2_penalty))(lstm_layer)

output_shape = y_train.shape[1]
lstm_units = 50
lambda_target = 0.01  
W_imprt = tf.Variable(tf.random.normal((input_shape[1], input_shape[1])), trainable=True)

print("Dimensiones de input_shape:", input_shape)
print("Dimensiones de output_shape:", output_shape)
print("Dimensiones de W_imprt:", W_imprt.shape)

model = build_model(input_shape=(1, 1), output_shape=1, lstm_units=100, learning_rate=learning_rate, l2_penalty=l2_penalty)

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Generar valores faltantes en la serie de tiempo de salida
missing_indices = np.random.choice(len(y_train), size=int(0.5 * len(y_train)), replace=False)
y_train_missing = np.copy(y_train)
y_train_missing = y_train_missing.astype(float)
y_train_missing[missing_indices] = np.nan 


print("Dimensiones de missing_indices:", missing_indices.shape)
print("Dimensiones de y_train_missing:", y_train_missing.shape)

# Imputar valores faltantes utilizando el RSV
RSV_train_missing = generate_RSV(model, X_train)

print("Dimensiones de RSV_train_missing:", RSV_train_missing.shape)
print("Dimensiones de W_imprt:", W_imprt.shape)
zt_1_train_missing = calculate_zt_1(W_imprt, RSV_train_missing)
print("Shape of zt_1_train_missing before squeezing:", zt_1_train_missing.shape)

y_train_imputed = np.copy(y_train_missing)
y_train_imputed[missing_indices] = np.expand_dims(zt_1_train_missing.numpy().flatten(), axis=1)[missing_indices]

#volver a entrenar si es necesario
#model.fit(X_train, y_train_imputed, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Calcular la pérdida total después de imputar valores faltantes
loss_imputed = Ltotal(zt_1_train_missing, X_train, np.ones_like(X_train), model.predict(X_train), y_train_missing, W_imprt, lambda_target)
print("Total Loss (after imputation):", loss_imputed.numpy())

y_train_imputed_descaled = scaler.inverse_transform(y_train_imputed)
y_train_descaled = scaler.inverse_transform(y_train)

mse_imputed = mean_squared_error(y_train_descaled, y_train_imputed_descaled)
print("Mean Squared Error (MSE) entre valores reales y valores imputados:", mse_imputed)

y_pred = model.predict(X_test)
y_pred = np.squeeze(y_pred, axis=1)


# Desescalar las predicciones y los valores reales
y_pred_descaled = scaler.inverse_transform(y_pred)
y_test_descaled = scaler.inverse_transform(y_test)


# Calcular el MSE entre las predicciones y los valores reales
mse = mean_squared_error(y_test_descaled, y_pred_descaled)
print("Mean Squared Error (MSE) en el conjunto de datos de prueba:", mse)

indices = missing_indices
data['date'] = pd.to_datetime(data['date'])
# Graficar los valores reales y los valores imputados
time_index = pd.date_range(start='1/11/2016 17:00', periods=len(y_train_descaled), freq='10min')


# Graficar los valores originales y los valores imputados en un gráfico de serie temporal
plt.figure(figsize=(10, 6))
plt.plot(time_index, y_train_descaled, label='Valores originales', color='blue')
plt.plot(time_index, y_train_imputed_descaled, label='Valores imputados', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.title('Comparación de valores originales y valores imputados')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(time_index, y_train_descaled, label='Valores originales', color='blue', marker='o')
plt.scatter(time_index, y_train_imputed_descaled, label='Valores imputados', color='red', marker='x')
plt.xlabel('Índice de muestra')
plt.ylabel('Valor')
plt.title('Comparación de valores originales y valores imputados')
plt.legend()
plt.show()