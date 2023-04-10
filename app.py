from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__,static_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    input_data1 = request.form.get("input_data1")
    i1 = np.array([float(val) for val in input_data1.split(',')])
    result = model.predict(i1.reshape(1, -1))

    if result[0] == 'R':
        result = 'The given object has been detected as a Rock'
    else:
        result = 'The given object has been detected as a Mine'

    return render_template('index.html', result=result,input_data1=input_data1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
