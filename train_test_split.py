import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados
df = pd.read_csv(r"C:\Users\jppec\Downloads\healthcare_dataset.csv")

# Verificar as primeiras linhas do DataFrame
print("Primeiras linhas do conjunto de dados:")
print(df.head())

# Verificar as colunas do DataFrame
print("\nColunas disponíveis no conjunto de dados:")
print(df.columns)

# Defina a variável de entrada (X) e a variável de saída (y)
X = df.drop(columns=['Medical Condition'])  # Variáveis de entrada (features)
y = df['Medical Condition']  # Variável de saída (target)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exibir as formas dos conjuntos de dados
print(f'\nForma do conjunto de treinamento: {X_train.shape}, {y_train.shape}')
print(f'Forma do conjunto de teste: {X_test.shape}, {y_test.shape}')

# Exibir informações sobre os dados
print("\nInformações sobre o conjunto de dados:")
print(df.info())
