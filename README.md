# Exploring Convolutional Layers Through Data and Experiments

## Problem Description
Explorar el efecto de decisiones arquitectónicas en capas convolucionales (CNN) usando un dataset real y experimentos controlados. Comparar contra un baseline sin convoluciones.

## Dataset
- **Fashion-MNIST**: imágenes 28x28 en escala de grises, 10 clases, tamaño manejable.
- Justificación: estructura espacial clara, adecuado para CNNs, ampliamente usado para docencia e investigación.

## Notebook
- `FashionMNIST_CNN_Exploration.ipynb` incluye:
  - EDA: tamaño, distribución de clases, ejemplos y normalización.
  - Baseline (Fully Connected): arquitectura, parámetros, entrenamiento y evaluación.
  - CNN: diseño intencional (2 conv, 3x3, padding same, pooling) y comparación.
  - Experimento controlado: kernel 3x3 vs 5x5.
  - Interpretación: sesgos inductivos, por qué funciona, cuándo no usar conv.
  - Despliegue en SageMaker: entrenamiento (`train.py`) y endpoint.

## Architecture (simple diagrams in text)
- **Baseline**: `Flatten -> Dense(128,relu) -> Dropout(0.2) -> Dense(10,softmax)`
- **CNN**: `Conv2D(32,3x3,same) -> MaxPool(2x2) -> Conv2D(64,3x3,same) -> MaxPool(2x2) -> Flatten -> Dense(64,relu) -> Dropout(0.3) -> Dense(10,softmax)`

## Experimental Results (expected)
- Baseline: menor accuracy por no capturar estructura espacial.
- CNN 3x3 vs 5x5: trade-off entre capacidad (capturar patrones más amplios) y complejidad. En Fashion-MNIST, 3x3 tiende a ser suficiente y más eficiente.

## Interpretation
- Convolución introduce localidad, compartición de pesos e invariancia a traslación.
- Beneficia tareas con estructura espacial en imágenes.
- No apropiada en datos tabulares o sin relación posicional clara.


## Setup
Instala dependencias:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
- Abre el notebook y ejecuta las celdas en orden.
- Al finalizar, ejecuta la última celda para guardar modelos y métricas en `models/` y `artifacts/`.
- Para SageMaker, configura variables de entorno `SAGEMAKER_EXECUTION_ROLE` y `S3_BUCKET` o edita directamente en la celda.



## Evidencia de SageMaker


<img width="1867" height="841" alt="image" src="https://github.com/user-attachments/assets/d3a7279c-dbc4-4d28-b79a-f486937e1e94" />
<img width="1914" height="738" alt="image" src="https://github.com/user-attachments/assets/07212a67-7db3-4524-9dd6-d97aacdb415a" />

- Curvas de entrenamiento/validación:
  
  <img width="1262" height="601" alt="image" src="https://github.com/user-attachments/assets/c8549295-f83c-4e87-8ae5-025f50cdc758" />


  
  <img width="1372" height="598" alt="image" src="https://github.com/user-attachments/assets/2ef5a4d0-052f-4b4d-bb29-1efc527b694b" />
<img width="1377" height="581" alt="image" src="https://github.com/user-attachments/assets/f8234c7b-e259-4d2d-9538-bc39f0fb686c" />

- Matriz de confusión (test, CNN):
  
  <img width="1070" height="412" alt="image" src="https://github.com/user-attachments/assets/0083547b-c0b5-4328-8f1f-bbd2159b5538" />
<img width="996" height="619" alt="image" src="https://github.com/user-attachments/assets/1a285610-957a-44ae-bc84-b27edfd34988" />

- Ejemplos de predicciones correctas e incorrectas:
  
  <img width="1485" height="821" alt="image" src="https://github.com/user-attachments/assets/5568fd2a-0570-4f3e-ab0e-2b3c2e44b0f6" />



## Conclusiones
- Rendimiento: TODO reemplazar con métricas de test (baseline vs CNN 3x3 vs 5x5).
- Arquitectura: la CNN 3x3 suele equilibrar capacidad y eficiencia en Fashion-MNIST; 5x5 puede captar patrones más amplios a costo de parámetros.
- Sesgo inductivo: la convolución introduce localidad, compartición de pesos e invariancia a traslación, beneficiando imágenes.
- Limitaciones: datos fuera del dominio visual o dependencias no locales podrían requerir otras arquitecturas (Transformers/GNNs).
- Despliegue: SageMaker permite escalar entrenamiento e inferencia; monitoreo y costos deben gestionarse.

## Explicación de lo hecho
- EDA: exploramos distribución de clases y ejemplos por clase, normalizamos a [0,1].
- Baseline: entrenamos una red densa para establecer referencia (accuracy/loss, train/val/test).
- CNN: diseñamos y entrenamos una arquitectura con 2 conv (3x3), pooling y capa densa.
- Experimentos: comparamos kernels 3x3 vs 5x5 manteniendo hiperparámetros fijos.
- Evaluación: graficamos curvas, calculamos métricas y guardamos modelos/métricas en `models/` y `artifacts/`.
- Reproducibilidad: fijamos semillas y generamos `requirements.txt`.
- Despliegue (pendiente): dejaremos scripts y evidencia de SageMaker en esta sección.
