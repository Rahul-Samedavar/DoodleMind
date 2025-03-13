from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall, TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adamax



#  Replace this with any version of interest: (Available: 20, 42, 44, 45, 46, 48, 50 )
version = 50
WEIGHTS_PATH = f"Weights/v_{version}.h5"


model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(35, activation='softmax') 
])
model.compile(
    optimizer=Adamax(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy',  TopKCategoricalAccuracy(3), Precision(), Recall()]
)

model.load_weights(WEIGHTS_PATH) 

app = Flask(__name__)

classes = ['Airplane', 'Alarm Clock', 'Ant', 'Bear', 'Beard', 'Bird', 'Bus',
       'Cookie', 'Cow', 'Donut', 'Hand', 'Hat', 'Key', 'Moon',
       'Motorbike', 'Octagon', 'Pizza', 'Rabbit', 'School Bus', 'Shark',
       'Skull', 'Smiley Face', 'Snake', 'Spider', 'Square', 'Star', 'Sun',
       'Swing Set', 'Table', 'Tent', 'Tree', 'Triangle', 'Whale', 'Wheel',
       'Windmill']

def label(pred):
    return {classes[i]: float(pred[0][i]) for i in range(len(classes))}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    doodle = request.get_json()['doodle']
    doodle = np.array(doodle)
    pred = model.predict(np.expand_dims(doodle, axis=0).astype(np.float16))[0].astype(np.float64)
    return {classes[i]: pred[i] for i in range(35)}
    
if __name__ == '__main__':
    app.run(debug=True)