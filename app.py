from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import bcrypt
import joblib  # Para carregar o modelo de IA
import numpy as np

app = Flask(__name__)

# Conectar ao MongoDB Atlas usando a Connection String
client = MongoClient("mongodb+srv://jppeccini:joao002233@cluster0.00xxv.mongodb.net/")
db = client["projeto_login"]  # Nome do banco de dados
users_collection = db["users"]  # Nome da coleção onde os usuários serão salvos

# Carregar o modelo de IA salvo
model = joblib.load('modelo_ia.pkl')  # Substitua pelo caminho correto do seu modelo

# Rota principal para servir a tela de login
@app.route('/')
def home():
    return render_template('login.html')

# Rota para registrar novos usuários
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data['email']
    senha = data['senha']

    # Verificar se o usuário já existe
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Usuário já registrado!"}), 400

    # Criptografar a senha
    hashed_senha = bcrypt.hashpw(senha.encode('utf-8'), bcrypt.gensalt())

    # Inserir no MongoDB
    users_collection.insert_one({
        "email": email,
        "senha": hashed_senha
    })

    return jsonify({"message": "Usuário registrado com sucesso!"}), 201

# Rota para realizar login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()  # Espera receber um JSON
    email = data.get('email')  # Acessa o e-mail do JSON
    senha = data.get('senha')  # Acessa a senha do JSON

    # Verificar se o usuário existe
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "Email não encontrado!"}), 400

    # Verificar se a senha está correta
    if bcrypt.checkpw(senha.encode('utf-8'), user['senha']):
        return jsonify({"message": "Login bem-sucedido!"}), 200
    else:
        return jsonify({"error": "Senha incorreta!"}), 400

# Rota para a tela de diagnóstico sem login
@app.route('/diagnostico_sem_login', methods=['GET', 'POST'])
def diagnostico_sem_login():
    if request.method == 'POST':
        data = request.get_json()
        local_dor = data.get('local_dor')
        tipo_sintoma = data.get('tipo_sintoma')
        nivel_dor = data.get('nivel_dor')

        # Pré-processamento para normalizar e verificar entradas
        locais_validos = {'cabeça': 0, 'ouvido': 1, 'olhos': 2, 'abdômen': 3}
        tipos_sintomas_validos = {'enxaqueca': 0, 'dengue': 1, 'otite': 2, 'cansaço': 3}

        # Verificar e converter valores
        local_dor = locais_validos.get(local_dor.lower(), -1)
        tipo_sintoma = tipos_sintomas_validos.get(tipo_sintoma.lower(), -1)

        try:
            nivel_dor = int(nivel_dor)
            nivel_dor = min(max(nivel_dor, 0), 10)  # Garantir que o nível de dor esteja entre 0 e 10
        except (ValueError, TypeError):
            nivel_dor = -1

        # Verificar valores inválidos
        if local_dor == -1 or tipo_sintoma == -1 or nivel_dor == -1:
            return jsonify({"error": "Dados de entrada inválidos."}), 400

        # Aqui deve haver a adição de mais variáveis se necessário.
        # Por exemplo, se você precisar de 11 features, você pode adicionar mais variáveis fictícias ou dados padrão.
        entrada = np.array([[local_dor, tipo_sintoma, nivel_dor] + [0]*(11-3)])  # Preenchendo com zeros para as demais features

        # Fazer a previsão usando o modelo de IA
        diagnostico = model.predict(entrada)

        # Enviar resposta com o diagnóstico
        return jsonify({
            "message": "Diagnóstico realizado com sucesso!",
            "diagnostico": diagnostico[0],
            "local_dor": data.get('local_dor'),
            "tipo_sintoma": data.get('tipo_sintoma'),
            "nivel_dor": data.get('nivel_dor')
        }), 200

    return render_template('diagnostico_sem_login.html')


# Iniciar o servidor Flask
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Alterado para porta 5001
