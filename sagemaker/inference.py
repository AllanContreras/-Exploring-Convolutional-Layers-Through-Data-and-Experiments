import os
import json
import numpy as np
import tensorflow as tf

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def model_fn(model_dir):
    """Load the trained model from the SageMaker model directory."""
    model_path = os.path.join(model_dir, 'model.keras')
    model = tf.keras.models.load_model(model_path)
    return model


def input_fn(request_body, content_type='application/json'):
    """Deserialize JSON or CSV input to a NumPy array suitable for the model."""
    if content_type == 'application/json':
        payload = json.loads(request_body)
        # Expect [[28x28], [28x28], ...]
        arr = np.array(payload, dtype=np.float32)
    elif content_type == 'text/csv':
        # CSV lines flattened to length 784
        arr = np.genfromtxt([request_body], delimiter=',', dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        arr = arr.reshape((-1, 28, 28))
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

    # Normalize and add channel dimension
    arr = arr / 255.0
    arr = np.expand_dims(arr, -1)
    return arr


def predict_fn(input_data, model):
    """Run prediction and return probabilities and predicted class."""
    probs = model.predict(input_data)
    preds = np.argmax(probs, axis=1).tolist()
    return {
        'probabilities': probs.tolist(),
        'predicted_index': preds,
        'predicted_label': [CLASS_NAMES[i] for i in preds]
    }


def output_fn(prediction, accept='application/json'):
    """Serialize prediction to the desired accept type."""
    if accept == 'application/json':
        return json.dumps(prediction)
    elif accept == 'text/csv':
        # Return only predicted indices in CSV
        return ','.join(map(str, prediction['predicted_index']))
    else:
        raise ValueError(f'Unsupported accept type: {accept}')
