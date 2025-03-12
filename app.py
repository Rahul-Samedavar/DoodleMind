from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model(r"Models\model_epoch_44.keras")

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
    # doodle = doodle.reshape((1, 28, 28, 1))
    # pred = model.predict(doodle)
    return {classes[i]: pred[i] for i in range(35)}
    
if __name__ == '__main__':
    app.run(debug=True)