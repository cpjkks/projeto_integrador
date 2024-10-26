import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar o dataset tratado
df = pd.read_csv('C:/Users/jppec/Downloads/healthcare_dataset/healthcare_dataset_tratado.csv')

# Converter colunas de data para datetime
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])

# Criar novas colunas a partir das datas (exemplo: dias entre as datas)
df['Days Admitted'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df.drop(columns=['Date of Admission', 'Discharge Date'], inplace=True)  # Remover colunas de data originais

# Definir variáveis de entrada e saída
X = df.drop(columns=['Medical Condition'])  # Variáveis de entrada
y = df['Medical Condition']  # Variável de saída

# Dividir o conjunto de dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo
model = RandomForestClassifier(random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
