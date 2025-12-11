import os
import io
import cv2
import numpy as np
import joblib
from PIL import Image
from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- CONFIGURAÇÕES ---
MODEL_PATH = 'models'
IMAGE_SIZE = (128, 128) # O mesmo tamanho usado no treino

# --- CARREGAMENTO DOS MODELOS E PRÉ-PROCESSADORES ---
# Carrega os modelos uma única vez quando a aplicação inicia para economizar tempo
try:
    # Carrega configurações de pré-processamento do SVM
    preprocess_config = joblib.load(os.path.join(MODEL_PATH, 'preprocess_config.pkl'))
    label_encoder = preprocess_config['label_encoder']
    scaler = preprocess_config['scaler']
    
    # Carrega o modelo SVM
    svm_model = joblib.load(os.path.join(MODEL_PATH, 'svm_model.pkl'))
    
    # Carrega o modelo YOLO
    yolo_model = YOLO(os.path.join(MODEL_PATH, 'best.pt'))
    
    # Mapeia os nomes originais do YOLO (das pastas) para os nomes traduzidos
    # A biblioteca YOLO usa os nomes das pastas como classes.
    yolo_class_names_original = yolo_model.names
    
    # Dicionário de tradução (o inverso do que usamos no treino)
    class_map_inverso = {
        "Limpo": "Clean",
        "Coberto de Neve": "Snow-Covered",
        "Sujeira de Pássaro": "Bird-drop",
        "Empoeirado": "Dusty",
        "Dano Elétrico": "Electrical-damage",
        "Dano Físico": "Physical-Damage"
    }

    # Criamos um dicionário de tradução para as classes do YOLO
    yolo_class_translation = {}
    for idx, name in yolo_class_names_original.items():
        for pt_name, en_name in class_map_inverso.items():
            if en_name == name:
                yolo_class_translation[idx] = pt_name
                break
    
    print("Modelos carregados com sucesso!")
    print("Classes do SVM (codificadas):", list(label_encoder.classes_))
    print("Classes do YOLO (traduzidas):", yolo_class_translation)

except Exception as e:
    print(f"Erro ao carregar os modelos: {e}")
    svm_model = None
    yolo_model = None

# --- FUNÇÃO DE PRÉ-PROCESSAMENTO PARA UMA ÚNICA IMAGEM ---
def preprocess_image_for_svm(image_bytes):
    # Converte bytes para uma imagem que o OpenCV possa usar
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)
    
    # Redimensiona e achata a imagem, assim como no treino
    resized_img = cv2.resize(image_np, IMAGE_SIZE)
    flattened_img = resized_img.flatten().reshape(1, -1)
    
    # Aplica o mesmo escalonamento do treino
    scaled_img = scaler.transform(flattened_img)
    return scaled_img

def preprocess_image_for_yolo(image_bytes):
    # O YOLO lida com o pré-processamento internamente,
    # então só precisamos passar o objeto de imagem PIL.
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image

# --- ROTAS DA API ---

@app.route('/', methods=['GET'])
def index():
    """Renderiza a página inicial."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recebe uma imagem, faz a predição com ambos os modelos e retorna o resultado."""
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nome de arquivo inválido'}), 400

    if file and svm_model and yolo_model:
        try:
            image_bytes = file.read()

            # --- PREDIÇÃO COM SVM ---
            processed_img_svm = preprocess_image_for_svm(image_bytes)
            svm_pred_encoded = svm_model.predict(processed_img_svm)
            svm_prediction = label_encoder.inverse_transform(svm_pred_encoded)[0]
            svm_confidence = np.max(svm_model.predict_proba(processed_img_svm)) * 100

            # --- PREDIÇÃO COM YOLO ---
            processed_img_yolo = preprocess_image_for_yolo(image_bytes)
            yolo_results = yolo_model(processed_img_yolo, verbose=False)
            
            # Pega o resultado principal (top 1)
            top1_result = yolo_results[0].probs.top1
            top1_confidence = yolo_results[0].probs.top1conf.item() * 100
            
            # Traduz o resultado do YOLO
            yolo_prediction = yolo_class_translation.get(top1_result, "Classe Desconhecida")

            # Monta a resposta
            result = {
                'svm': {
                    'class': svm_prediction,
                    'confidence': f"{svm_confidence:.2f}%"
                },
                'yolo': {
                    'class': yolo_prediction,
                    'confidence': f"{top1_confidence:.2f}%"
                }
            }
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'Erro durante a predição: {str(e)}'}), 500
    
    return jsonify({'error': 'Modelos não carregados ou arquivo inválido'}), 500

if __name__ == '__main__':
    # Roda o servidor Flask em modo de depuração
    app.run(debug=True, host='0.0.0.0', port=5000)