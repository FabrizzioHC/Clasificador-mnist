import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image

# DATASET CARGADO DESDE LA NUBE
@st.cache_resource
def cargar_y_entrenar():
    # Descarga el dataset directo desde los servidores de TensorFlow en internet
    (x_train, y_train), (x_test, y_test) = mnist.load_data() #esto de aca descarga desde servidores de tensorflow

    # Usamos 20000 muestras
    x_train = x_train[:20000] / 255.0
    y_train = to_categorical(y_train[:20000], 10)
    x_test = x_test[:1000] / 255.0
    y_test = to_categorical(y_test[:1000], 10)

    # Modelo simple
    modelo = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    modelo.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    modelo.fit(x_train, y_train, epochs=10, verbose=0)
    return modelo, x_test, y_test

# INTERFAZ CON STREAMLIT
st.title("Clasificador de Dígitos MNIST")
st.write("Versión 3 - Dataset cargado desde la nube")
st.write("El modelo se entrena con datos descargados automáticamente desde internet.")

st.info("Descargando dataset desde la nube y entrenando...")
modelo, x_test, y_test = cargar_y_entrenar()
st.success("Modelo entrenado correctamente")

# Mostrar precisión
loss, acc = modelo.evaluate(x_test, y_test, verbose=0)
st.metric("Precisión del modelo", f"{acc*100:.2f}%")

# Subir imagen para predecir
st.divider()
st.subheader("Sube una imagen de un dígito")
archivo = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

if archivo is not None:
    imagen = Image.open(archivo).convert("L").resize((28, 28))
    st.image(imagen, caption="Imagen cargada", width=150)

    arr = np.array(imagen) / 255.0
    arr = arr.reshape(1, 28, 28)

    prediccion = modelo.predict(arr)
    digito = np.argmax(prediccion)
    confianza = np.max(prediccion) * 100

    st.success(f"El modelo predice que es el dígito: **{digito}**")
    st.write(f"Confianza: {confianza:.2f}%")