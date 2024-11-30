from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Inicializando o app Flask
app = Flask(__name__)

# Caminho para o banco de dados
DATABASE = "banco_dados.db"

# Carregando o modelo e os LabelEncoders
model = joblib.load('modelo_sintoma_diagnostico.pkl')
le_sintoma = joblib.load('label_encoder_sintoma.pkl')
le_parte_corpo = joblib.load('label_encoder_parte_corpo.pkl')
le_diagnostico = joblib.load('label_encoder_diagnostico.pkl')

# Função auxiliar para conectar ao banco de dados
def conectar_bd():
    return sqlite3.connect(DATABASE)

# Função para treinar o modelo
def treinar_modelo():
    # Conectando ao banco de dados para pegar os sintomas
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("SELECT sintoma, parte_corpo, intensidade, diagnostico FROM sintomas")
    dados = cursor.fetchall()
    conn.close()

    # Criando um DataFrame a partir dos dados do banco
    df = pd.DataFrame(dados, columns=['sintoma', 'parte_corpo', 'intensidade', 'diagnostico'])

    # Atualizando os LabelEncoders com todos os diagnósticos presentes no banco de dados
    global le_sintoma, le_parte_corpo, le_diagnostico
    le_sintoma = joblib.load('label_encoder_sintoma.pkl')
    le_parte_corpo = joblib.load('label_encoder_parte_corpo.pkl')
    le_diagnostico = joblib.load('label_encoder_diagnostico.pkl')

    sintoma_encoded = le_sintoma.fit_transform(df['sintoma'])
    parte_corpo_encoded = le_parte_corpo.fit_transform(df['parte_corpo'])
    diagnostico_encoded = le_diagnostico.fit_transform(df['diagnostico'])

    # Treinando o modelo RandomForest
    X = pd.DataFrame({
        'sintoma': sintoma_encoded,
        'parte_corpo': parte_corpo_encoded,
        'intensidade': df['intensidade']
    })
    y = diagnostico_encoded

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Salvando o modelo e os LabelEncoders atualizados
    joblib.dump(model, 'modelo_sintoma_diagnostico.pkl')
    joblib.dump(le_sintoma, 'label_encoder_sintoma.pkl')
    joblib.dump(le_parte_corpo, 'label_encoder_parte_corpo.pkl')
    joblib.dump(le_diagnostico, 'label_encoder_diagnostico.pkl')

# Rota para exibir a tela de login
@app.route('/')
def login_page():
    return render_template('login.html')

# Rota para a página de usuário sem login
@app.route('/diagnostico_sem_login')
def usuario_sem_login():
    return render_template('usuario_sem_login.html')

# Rota para inserir sintomas no banco (Usuário Sem Login)
@app.route('/api/sintomas', methods=['POST'])
def salvar_sintoma():
    dados = request.json
    sintoma = dados.get('sintoma')
    parte_corpo = dados.get('parteCorpo')
    intensidade = dados.get('intensidade')

    if not sintoma or not parte_corpo or intensidade is None:
        return jsonify({"error": "Dados incompletos"}), 400

    # Verificar se o sintoma já foi diagnosticado
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute(""" 
        SELECT id, sintoma, parte_corpo, intensidade, diagnostico
        FROM sintomas 
        WHERE sintoma = ? AND parte_corpo = ? AND intensidade = ?
    """, (sintoma, parte_corpo, intensidade))
    resultado = cursor.fetchone()

    if resultado:
        # Sintoma já foi diagnosticado, retorna o diagnóstico
        id_, sintoma, parte_corpo, intensidade, diagnostico = resultado
        return jsonify({"message": f"Sintoma já diagnosticado: {sintoma} - Diagnóstico: {diagnostico}"}), 200

    # Se o sintoma não foi diagnosticado, insere no banco
    cursor.execute("""
        INSERT INTO sintomas (sintoma, parte_corpo, intensidade, diagnostico)
        VALUES (?, ?, ?, ?)
    """, (sintoma, parte_corpo, intensidade, ''))
    conn.commit()

    # Agora que o sintoma foi salvo, podemos fazer a previsão
    try:
        sintoma_encoded = le_sintoma.transform([sintoma])[0]
        parte_corpo_encoded = le_parte_corpo.transform([parte_corpo])[0]
    except Exception as e:
        return jsonify({"error": f"Erro na codificação: {e}"}), 400

    # Fazendo a previsão
    predicao = model.predict([[sintoma_encoded, parte_corpo_encoded, intensidade]])

    # Decodificando o resultado
    diagnostico_predito = le_diagnostico.inverse_transform(predicao)[0]

    # Atualizando o banco com o diagnóstico
    cursor.execute("""
        UPDATE sintomas
        SET diagnostico = ?
        WHERE sintoma = ? AND parte_corpo = ? AND intensidade = ?
    """, (diagnostico_predito, sintoma, parte_corpo, intensidade))
    conn.commit()
    conn.close()

    # Re-treinando o modelo após inserção dos dados
    treinar_modelo()

    return jsonify({"message": f"Sintoma registrado com sucesso! Diagnóstico: {diagnostico_predito}"}), 201

# Rota para listar sintomas (Administrador)
@app.route('/api/sintomas', methods=['GET'])
def listar_sintomas():
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("SELECT id, sintoma, parte_corpo, intensidade, diagnostico FROM sintomas")
    sintomas = cursor.fetchall()
    conn.close()

    resultado = [
        {"id": id_, "sintoma": sintoma, "parteCorpo": parte_corpo, "intensidade": intensidade, "diagnostico": diagnostico}
        for id_, sintoma, parte_corpo, intensidade, diagnostico in sintomas
    ]
    return jsonify(resultado)

# Rota para previsão de diagnóstico (Usuário Sem Login)
@app.route('/api/prever_diagnostico', methods=['POST'])
def prever_diagnostico():
    dados = request.json
    sintoma = dados.get('sintoma')
    parte_corpo = dados.get('parteCorpo')
    intensidade = dados.get('intensidade')

    if not sintoma or not parte_corpo or intensidade is None:
        return jsonify({"error": "Dados incompletos"}), 400

    # Codificando os dados de entrada
    try:
        sintoma_encoded = le_sintoma.transform([sintoma])[0]
        parte_corpo_encoded = le_parte_corpo.transform([parte_corpo])[0]
    except Exception as e:
        return jsonify({"error": f"Erro na codificação: {e}"}), 400

    # Fazendo a previsão
    predicao = model.predict([[sintoma_encoded, parte_corpo_encoded, intensidade]])

    # Tentando decodificar o resultado, e tratando erro se necessário
    try:
        diagnostico_predito = le_diagnostico.inverse_transform(predicao)[0]
    except ValueError as e:
        return jsonify({"error": f"Erro ao decodificar diagnóstico: {e}"}), 400

    return jsonify({"diagnostico": diagnostico_predito})

# Rota para painel de administração
@app.route('/admin_painel')
def admin_painel():
    return render_template('admin_painel.html')

# Rota para adicionar novos dados e treinar o modelo
@app.route('/api/adicionar_dados', methods=['POST'])
def adicionar_dados():
    dados = request.json
    sintoma = dados.get('sintoma')
    parte_corpo = dados.get('parteCorpo')
    intensidade = dados.get('intensidade')
    diagnostico = dados.get('diagnostico')

    if not sintoma or not parte_corpo or intensidade is None or not diagnostico:
        return jsonify({"error": "Dados incompletos"}), 400

    # Inserir os novos dados no banco de dados
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sintomas (sintoma, parte_corpo, intensidade, diagnostico)
        VALUES (?, ?, ?, ?)
    """, (sintoma, parte_corpo, intensidade, diagnostico))
    conn.commit()
    conn.close()

    # Re-treinando o modelo após inserção dos dados
    treinar_modelo()

    return jsonify({"message": "Dados inseridos com sucesso e modelo treinado!"}), 201

# Inicializar o servidor
if __name__ == "__main__":
    if not os.path.exists(DATABASE):
        print("Banco de dados não encontrado. Execute o script create_db.sql para criar.")
    app.run(debug=True)
