import pandas as pd

df = pd.read_excel('Resultados\Resultados_Finais.xlsx')

columns_order = ['Base','Tipo','Loja','Item','Modelo','RMSE(#)','MAPE(%)','Treino','Previsao']

df = df[columns_order]

df.sort_values(by=columns_order[0:5], ascending=True, inplace=True)

nomes = {
    'Base': 'Base',
    'Tipo': 'Tipo',
    'Loja': 'Loja',
    'Item': 'Produto',
    'Modelo': 'Modelo',
    'RMSE(#)': 'RMSE',
    'MAPE(%)': 'MAPE',
    'Treino': 'Treino',
    'Previsao': 'Previsao'
}

df.rename(columns=nomes, inplace=True)

df['Modelo'].replace('LSMT','LSTM', inplace=True)
df['Tipo'].replace('Item','Produto', inplace=True)
df['Treino'] = df['Treino'].round(1)
df['Previsao'] = df['Previsao'].round(2)
df['RMSE'] = df['RMSE'].astype(int)
df['Série Temporal'] = df['Base'] + '_' + df['Tipo'] + '_' + df['Loja'].astype(str) + '_' + df['Produto'].astype(str) + '_' + df['Modelo'].astype(str)
df.drop(columns=['Base','Tipo','Loja','Produto','Modelo'], inplace=True)
columns_order = ['Série Temporal','RMSE','MAPE','Treino','Previsao']
df = df[columns_order]
df['MAPE'] = df['MAPE'].round(1)

def formatar_numero(x):
    if abs(x) < 1000:  # Mantém números com até 3 dígitos sem notação científica
        return str(round(x, 1))  # Arredonda para 1 casa decimal
    else:
        return f"{x:.1e}"  # Formata em notação científica com 1 casa decimal

# Aplicando a formatação
df['MAPE'] = df['MAPE'].apply(formatar_numero)

df.tail(60)
df.to_csv('teste2.csv', index=False)
df.to_excel('teste.xlsx', index=False)