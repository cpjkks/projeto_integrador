import pandas as pd

# Carregar o dataset
df = pd.read_excel('C:/Users/jppec/Downloads/healthcare_dataset/healthcare_dataset.xlsx')

# Exibir as primeiras linhas do dataframe para verificação
print("Primeiras linhas do dataframe:")
print(df.head())

# Remover colunas que não são necessárias
# Ajuste a lista de colunas a serem removidas de acordo com o que você tem
df = df.drop(columns=['Gender', 'Blood Type', 'Admission Type', 'Medication', 'Test Results'])

# Tratar colunas textuais (se necessário, dependendo dos dados)
# Exemplo de tratamento: converter todas as strings para minúsculas
df['Medical Condition'] = df['Medical Condition'].str.lower().str.strip()

# Salvar o dataset tratado para evitar erros posteriores
df.to_csv('C:/Users/jppec/Downloads/healthcare_dataset/healthcare_dataset_tratado.csv', index=False)

print("Dataset tratado e salvo com sucesso!")
