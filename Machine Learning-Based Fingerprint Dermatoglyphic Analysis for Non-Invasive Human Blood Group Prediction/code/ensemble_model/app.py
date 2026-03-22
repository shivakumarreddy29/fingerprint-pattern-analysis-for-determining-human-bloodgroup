import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'ensemble_best_model.h5'
class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# --- Model Architecture Definitons (Rebuild to avoid load_model errors) ---
def build_alexnet(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same')(inp)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Model(inputs=inp, outputs=x, name='AlexNet')

def build_lenet(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(6, (5, 5), activation='tanh', padding='same')(inp)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(16, (5, 5), activation='tanh', padding='valid')(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation='tanh')(x)
    x = Dense(84, activation='tanh')(x)
    return Model(inputs=inp, outputs=x, name='LeNet')

def build_resnet(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    return Model(inputs=base_model.input, outputs=x, name='ResNet50')

def build_ensemble(input_shape, num_classes):
    alexnet = build_alexnet(input_shape)
    lenet = build_lenet(input_shape)
    resnet = build_resnet(input_shape)
    
    model_input = Input(shape=input_shape)
    
    # We need to call the models on the input tensor
    out_alex = alexnet(model_input)
    out_lenet = lenet(model_input)
    out_res = resnet(model_input)
    
    combined = Concatenate()([out_alex, out_lenet, out_res])
    z = Dense(256, activation='relu')(combined)
    z = Dropout(0.5)(z)
    output = Dense(num_classes, activation='softmax')(z)
    return Model(inputs=model_input, outputs=output, name='Ensemble_Model')

# Initialize model globally
model = None
try:
    print("Building model architecture...")
    model = build_ensemble((224, 224, 3), 8)
    print(f"Loading weights from {MODEL_PATH}...")
    model.load_weights(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model weights: {e}")
    # We continue so the app starts, but prediction will fail gracefully

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file:
        try:
            # Process image
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Rescale
            
            if model:
                predictions = model.predict(img_array)
                class_idx = np.argmax(predictions, axis=1)[0]
                confidence = predictions[0][class_idx]
                prediction_label = class_names[class_idx]
                
                return render_template('index.html', 
                                     prediction=prediction_label, 
                                     confidence=f"{confidence:.2%}",
                                     image_data=None) # Could pass base64 image here if needed
            else:
                return render_template('index.html', error='Model not loaded. Check server logs.')
                
        except Exception as e:
            return render_template('index.html', error=f'Error processing image: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
