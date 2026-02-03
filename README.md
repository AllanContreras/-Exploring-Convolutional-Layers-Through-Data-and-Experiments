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

## SageMaker
- Usa SDK de SageMaker con `train.py` empaquetado.
- Requiere `role` IAM, `bucket` S3 y credenciales AWS configuradas.

## Setup
Instala dependencias:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
- Abre el notebook y ejecuta las celdas en orden.
- Para SageMaker, configura variables de entorno `SAGEMAKER_EXECUTION_ROLE` y `S3_BUCKET` o edita directamente en la celda.

## Optional (bonus)
- Visualiza filtros/feature maps con capas intermedias.
