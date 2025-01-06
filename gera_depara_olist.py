import pandas as pd

# Função para gerar mapeamento automático
def generate_mapping(df, column_name, prefix):
    unique_values = df[column_name].unique()
    return {value: f"{prefix}_{i+1}" for i, value in enumerate(unique_values)}

# Aplicar o mapeamento aos dataframes
def apply_mapping(df, loja_mapping, item_mapping=None):
    df['Loja'] = df['Loja'].map(loja_mapping)
    if item_mapping:
        df['Item'] = df['Item'].map(item_mapping)
    return df

# Gerar mapeamento automático para lojas e itens
df_sample = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\CNN.csv')
loja_mapping = generate_mapping(df_sample, 'Loja', 'Loja')
item_mapping = generate_mapping(df_sample, 'Item', 'Item')

# ITEM

arima_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\ARIMA.csv')
cnn_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\CNN.csv')
lsmt_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\LSMT.csv')
prophet_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\prophet.csv')
rl_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\RL.csv')
arima_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\ARIMA.csv')
cnn_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\CNN.csv')
lsmt_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\LSMT.csv')
prophet_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\prophet.csv')
rl_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\RL.csv')


arima_df_olist = apply_mapping(arima_df_olist, loja_mapping, item_mapping)
cnn_df_olist = apply_mapping(cnn_df_olist, loja_mapping, item_mapping)
lsmt_df_olist = apply_mapping(lsmt_df_olist, loja_mapping, item_mapping)
prophet_df_olist = apply_mapping(prophet_df_olist, loja_mapping, item_mapping)
rl_df_olist = apply_mapping(rl_df_olist, loja_mapping, item_mapping)

arima_df_olist_tempo = apply_mapping(arima_df_olist_tempo, loja_mapping, item_mapping)
cnn_df_olist_tempo = apply_mapping(cnn_df_olist_tempo, loja_mapping, item_mapping)
lsmt_df_olist_tempo = apply_mapping(lsmt_df_olist_tempo, loja_mapping, item_mapping)
prophet_df_olist_tempo = apply_mapping(prophet_df_olist_tempo, loja_mapping, item_mapping)
rl_df_olist_tempo = apply_mapping(rl_df_olist_tempo, loja_mapping, item_mapping)

arima_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\ARIMA.csv', index=False)
cnn_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\CNN.csv', index=False)
lsmt_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\LSMT.csv', index=False)
prophet_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\prophet.csv', index=False)
rl_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Item\RL.csv', index=False)

arima_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\ARIMA.csv', index=False)
cnn_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\CNN.csv', index=False)
lsmt_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\LSMT.csv', index=False)
prophet_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\prophet.csv', index=False)
rl_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Item\RL.csv', index=False)



# LOJA

arima_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\ARIMA.csv')
cnn_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\CNN.csv')
lsmt_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\LSMT.csv')
prophet_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\prophet.csv')
rl_df_olist = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\RL.csv')
arima_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\ARIMA.csv')
cnn_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\CNN.csv')
lsmt_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\LSMT.csv')
prophet_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\prophet.csv')
rl_df_olist_tempo = pd.read_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\RL.csv')

arima_df_olist = apply_mapping(arima_df_olist, loja_mapping)
cnn_df_olist = apply_mapping(cnn_df_olist, loja_mapping)
lsmt_df_olist = apply_mapping(lsmt_df_olist, loja_mapping)
prophet_df_olist = apply_mapping(prophet_df_olist, loja_mapping)
rl_df_olist = apply_mapping(rl_df_olist, loja_mapping)

arima_df_olist_tempo = apply_mapping(arima_df_olist_tempo, loja_mapping)
cnn_df_olist_tempo = apply_mapping(cnn_df_olist_tempo, loja_mapping)
lsmt_df_olist_tempo = apply_mapping(lsmt_df_olist_tempo, loja_mapping)
prophet_df_olist_tempo = apply_mapping(prophet_df_olist_tempo, loja_mapping)
rl_df_olist_tempo = apply_mapping(rl_df_olist_tempo, loja_mapping)

arima_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\ARIMA.csv', index=False)
cnn_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\CNN.csv', index=False)
lsmt_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\LSMT.csv', index=False)
prophet_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\prophet.csv', index=False) 
rl_df_olist.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Previsao\Olist\Loja\RL.csv', index=False)

arima_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\ARIMA.csv', index=False)
cnn_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\CNN.csv', index=False)
lsmt_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\LSMT.csv', index=False)
prophet_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\prophet.csv', index=False)
rl_df_olist_tempo.to_csv(r'C:\Users\BRAINER.CAMPOS\Desktop\TCC\Resultados\Tempo\Olist\Loja\RL.csv', index=False)
