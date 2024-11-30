import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import sqlite3

# Caminho do arquivo CSV (dados iniciais) e do banco de dados
DATASET_PATH = r"C:\Users\jppec\Downloads\dados_sintomas_10000_sem_acentos.csv"
DATABASE_PATH = 'banco_dados.db'

def conectar_bd():
    return sqlite3.connect(DATABASE_PATH)

def train_model():
    # Carregando dados iniciais do CSV
    try:
        df = pd.read_csv(DATASET_PATH, sep=',')  # Certifique-se de que o separador seja vírgula
    except Exception as e:
        print(f"Erro ao carregar o arquivo CSV: {e}")
        return

    # Verificar se as colunas esperadas existem no arquivo CSV
    colunas_esperadas = ['sintoma', 'parte_corpo', 'intensidade', 'diagnostico']
    if not all(col in df.columns for col in colunas_esperadas):
        raise ValueError(f"Colunas esperadas não encontradas no arquivo. Esperado: {colunas_esperadas}, Encontrado: {list(df.columns)}")

    # Inicializando os LabelEncoders para cada coluna categórica
    le_sintoma = LabelEncoder()
    le_parte_corpo = LabelEncoder()
    le_diagnostico = LabelEncoder()

    # Codificando as colunas categóricas
    df['sintoma_encoded'] = le_sintoma.fit_transform(df['sintoma'])
    df['parte_corpo_encoded'] = le_parte_corpo.fit_transform(df['parte_corpo'])
    df['diagnostico_encoded'] = le_diagnostico.fit_transform(df['diagnostico'])

    # Definindo os dados de entrada (X) e a saída (y)
    X = df[['sintoma_encoded', 'parte_corpo_encoded', 'intensidade']]
    y = df['diagnostico_encoded']

    # Criando e treinando o modelo RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Salvando o modelo e os codificadores
    joblib.dump(model, 'modelo_sintoma_diagnostico.pkl')
    joblib.dump(le_sintoma, 'label_encoder_sintoma.pkl')
    joblib.dump(le_parte_corpo, 'label_encoder_parte_corpo.pkl')
    joblib.dump(le_diagnostico, 'label_encoder_diagnostico.pkl')

    print("Modelo inicial treinado e salvo com sucesso!")

    # Agora vamos buscar os dados do banco de dados para o treinamento contínuo
    conn = conectar_bd()
    cursor = conn.cursor()

    # Buscando os dados da tabela sintomas (dados inseridos após o treinamento inicial)
    cursor.execute("SELECT sintoma, parte_corpo, intensidade, diagnostico FROM sintomas")
    dados = cursor.fetchall()
    conn.close()

    if not dados:
        print("Nenhum dado adicional encontrado no banco de dados para treinamento contínuo.")
        return

    # Criando um DataFrame com os dados do banco
    df_banco = pd.DataFrame(dados, columns=['sintoma', 'parte_corpo', 'intensidade', 'diagnostico'])

    # Codificando novamente as colunas do banco de dados com os mesmos LabelEncoders
    df_banco['sintoma_encoded'] = le_sintoma.transform(df_banco['sintoma'])
    df_banco['parte_corpo_encoded'] = le_parte_corpo.transform(df_banco['parte_corpo'])
    df_banco['diagnostico_encoded'] = le_diagnostico.transform(df_banco['diagnostico'])

    # Definindo os dados de entrada (X) e a saída (y)
    X_banco = df_banco[['sintoma_encoded', 'parte_corpo_encoded', 'intensidade']]
    y_banco = df_banco['diagnostico_encoded']

    # Criando e treinando o modelo com os dados do banco
    model.fit(X_banco, y_banco)

    # Salvando novamente o modelo e os codificadores
    joblib.dump(model, 'modelo_sintoma_diagnostico.pkl')
    print("Modelo atualizado com os dados do banco de dados!")

if __name__ == "__main__":
    train_model()
