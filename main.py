import numpy as np
import sounddevice as sd
import time
import tensorflow as tf

MODEL_FILENAME = "model/soundclassifier_with_metadata.tflite"
LABELS_FILENAME = "model/labels.txt"

labels = []
with open(LABELS_FILENAME, 'r', encoding='utf-8') as f:
    for line in f:
        labels.append(line.strip().split(' ')[1])

print(f"Classes carregadas: {labels}")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILENAME)
    interpreter.allocate_tensors()
except ValueError:
    print(f"Erro: Não foi possível carregar o modelo '{MODEL_FILENAME}'.")
    print("Verifique se o arquivo está na pasta correta e foi baixado corretamente.")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Modelo TFLite carregado com sucesso.")

try:
    # A entrada do TM Audio é [batch_size, num_samples]
    AUDIO_LENGTH = input_details[0]['shape'][1]
    print(f"Tamanho da entrada do modelo: {AUDIO_LENGTH} amostras")
except (IndexError, TypeError):
    AUDIO_LENGTH = 44032
    print(f"Usando tamanho de entrada padrão: {AUDIO_LENGTH} amostras")

# O tipo de dado do modelo (geralmente float32)
INPUT_DTYPE = input_details[0]['dtype']
SAMPLE_RATE = 44100
CHANNELS = 1

# Limiar de confiança
CONFIDENCE_THRESHOLD = 0.75

print("\nIniciando reconhecimento de som (TFLite)...")
print("Pressione Ctrl+C para parar.")

try:
    while True:
        print("Ouvindo...")

        # Grava um bloco de áudio do tamanho que o modelo espera
        my_recording = sd.rec(int(AUDIO_LENGTH),
                              samplerate=SAMPLE_RATE,
                              channels=CHANNELS,
                              dtype=INPUT_DTYPE)
        sd.wait()

        # O formato de entrada do TFLite é (1, AUDIO_LENGTH)
        # O sd.rec() nos dá (AUDIO_LENGTH, 1).

        # Achata o áudio para (AUDIO_LENGTH,)
        flattened_recording = my_recording.flatten()

        # Adiciona a dimensão do batch para (1, AUDIO_LENGTH)
        data_to_predict = np.expand_dims(flattened_recording, axis=0)

        # Define o tensor de entrada
        interpreter.set_tensor(input_details[0]['index'], data_to_predict)

        # Executa a inferência
        interpreter.invoke()

        # Pega o tensor de saída (resultado)
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Obtem o índice da classe com maior probabilidade
        predicted_index = np.argmax(prediction[0])

        # Obtem a probabilidade dessa classe
        confidence = prediction[0][predicted_index]

        # Obtem o nome da classe
        predicted_label = labels[predicted_index]

        print("\n--- Resultado ---")
        if confidence > CONFIDENCE_THRESHOLD:
            print(f"Predição: {predicted_label}")
            print(f"Confiança: {confidence:.2%}")
        else:
            print(f"Não tenho certeza (confiança baixa: {confidence:.2%})")

        print("\nProbabilidades:")
        for i, label in enumerate(labels):
            print(f"  {label}: {prediction[0][i]:.2%}")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nEncerrando...")
