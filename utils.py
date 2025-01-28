import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from prophet import Prophet
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Função para pré-processamento dos dados
def preprocess_data_LSTM(data, train_size_ratio=0.7):
    """
    Pré-processa os dados para treinamento e teste.
    
    Parâmetros:
        data (array): Série temporal (valores).
        train_size_ratio (float): Proporção do conjunto de treinamento.
    
    Retorna:
        train, test (arrays): Dados divididos em treinamento e teste.
        scaler (MinMaxScaler): Objeto scaler usado para normalização.
    """
    data = data.reshape(-1, 1)
    
    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    # Determinar os tamanhos dos conjuntos
    train_size = int(len(data) * train_size_ratio)
    
    # Divisão em treinamento e teste
    train = data[:train_size]
    test = data[train_size:]
    
    return train, test, scaler


# Função para criar o dataset no formato necessário para LSTM
def create_dataset_LSTM(dataset, look_back=1):
    """
    Cria entradas e saídas formatadas para modelos LSTM.
    
    Parâmetros:
        dataset (ndarray): Dados normalizados.
        look_back (int): Número de timesteps para usar como entrada.
    
    Retorna:
        X (ndarray): Dados de entrada no formato (amostras, timesteps, features).
        Y (ndarray): Dados de saída correspondentes.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # Criar sequência de entrada e a saída correspondente
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    
    # Converter listas para arrays numpy
    X = np.array(X)
    Y = np.array(Y)
    
    # Redimensionar X para (amostras, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, Y


def create_and_train_model_LSTM(X_train, y_train, look_back, epochs=100, batch_size=32):
    """
    Cria, compila e treina um modelo LSTM para previsão de séries temporais.
    
    Parâmetros:
        X_train, y_train: Dados de treinamento.
        look_back: Número de passos para previsão.
        epochs: Número de épocas.
        batch_size: Tamanho do lote.
    
    Retorna:
        model (Sequential): Modelo treinado.
        history (History): Histórico do treinamento.
    """
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    
    history = model.fit(
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=[early_stopping], 
        verbose=0  # Set verbose to 0 to not display epochs
    )
    
    return model, history


def prepare_data_prophet(walmart_data):
    # Renomear colunas
    walmart_data = walmart_data.rename(columns={'Data': 'ds', 'Vendas': 'y'})
    return walmart_data

def split_data_prophet(df, train_ratio=0.7):
    train_size = int(len(df) * train_ratio)
    test_size = len(df) - train_size

    train = df[:train_size]
    test = df[train_size:]
    
    return train, test

def train_model_prophet(train):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,  # Ajuste de sensibilidade a mudanças
        interval_width=0.95  # Largura do intervalo de confiança
    )
    #model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(train)
    return model

def make_forecast_prophet(model, periods, freq='W'):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_metrics(df_real, df_pred):
    rmse = calculate_rmse(df_real, df_pred)
    mape = calculate_mape(df_real, df_pred)
    return rmse, mape

def prepare_data_CNN(data_frame):
    data = data_frame.values  # Substitua 'favorita_agrupado' pelo seu DataFrame
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def create_sequences_CNN(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length,0]
        y = data[i + seq_length,0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def split_data_CNN(data_scaled, seq_length, train_ratio=0.7):
    train_size = int(len(data_scaled) * train_ratio)  # 70% treino, 30% teste
    X, y = create_sequences_CNN(data_scaled, seq_length)
    
    # Ajuste para evitar buracos: o conjunto de teste começa imediatamente após o treino
    X_train, X_test = X[:train_size], X[train_size - seq_length:]
    y_train, y_test = y[:train_size], y[train_size - seq_length:]
    
    return X_train, X_test, y_train, y_test, train_size


def build_model_CNN(seq_length):
    model = tf.keras.Sequential([
        # Primeira camada Conv1D
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                               kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(seq_length, 1)),
        tf.keras.layers.BatchNormalization(),  # Normalização
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Segunda camada Conv1D
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                               kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),  # Normalização
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Camada Flatten
        tf.keras.layers.Flatten(),
        
        # Primeira camada totalmente conectada (Dense)
        tf.keras.layers.Dense(64, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),  # Normalização
        tf.keras.layers.Dropout(0.3),
        
        # Saída
        tf.keras.layers.Dense(1)
    ])
    return model


def train_model_CNN(model, X_train, y_train, epochs=200, batch_size=32):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=0)
    return history

def make_predictions_CNN(model, X_test):
    return model.predict(X_test)

