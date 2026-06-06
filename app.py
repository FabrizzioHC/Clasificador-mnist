import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os

# ENTRENAR SOLO SI EL MODELO NO EXISTE
@st.cache_resource
def cargar_modelo():

    if not os.path.exists("modelo_mnist.keras"):

        st.warning("Entrenando modelo por primera vez...")

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train[:20000] / 255.0
        y_train = to_categorical(y_train[:20000], 10)

        modelo = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        modelo.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        modelo.fit(x_train, y_train, epochs=10, verbose=0)

        modelo.save("modelo_mnist.keras")

    return load_model("modelo_mnist.keras")


st.title("Clasificador de Dígitos MNIST")
st.write("Versión optimizada para Render")

modelo = cargar_modelo()

st.success("Modelo cargado correctamente")
st.divider()
st.subheader("Sube una imagen de un dígito")

archivo = st.file_uploader(
    "Selecciona una imagen",
    type=["png", "jpg", "jpeg"]
)

if archivo is not None:

    imagen = Image.open(archivo).convert("L").resize((28, 28))

    st.image(imagen, caption="Imagen cargada", width=150)

    arr = np.array(imagen) / 255.0
    arr = arr.reshape(1, 28, 28)

    prediccion = modelo.predict(arr)

    digito = np.argmax(prediccion)
    confianza = np.max(prediccion) * 100

    st.success(
        f"El modelo predice que es el dígito: **{digito}**"
    )

    st.write(f"Confianza: {confianza:.2f}%")