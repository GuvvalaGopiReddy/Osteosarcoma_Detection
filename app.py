from flask import Flask, render_template, request, send_from_directory
import random, os
from werkzeug.utils import secure_filename


app = Flask(__name__)
random.seed(0)
app.config['SECRET_KEY'] = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

dir_path = os.path.dirname(os.path.realpath(__file__))

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
	return render_template('result.html')

@app.route('/algo', methods=['GET', 'POST'])
def algo():
	return render_template('algo.html')



import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Define the prediction function
def predict_image_class(image_path, model_path='/kaggle/working/trained_modelvgg.h5'):
    """
    Predicts the class of an input image using a trained VGG16 model.

    Args:
        image_path (str): Path to the image.
        model_path (str): Path to the trained model.

    Returns:
        dict: A dictionary containing the predicted class and class probabilities.
    """
    # Class labels
    class_labels = ['Non-Tumor', 'Non-Viable-Tumor', 'Viable']

    # Load the trained model
    model = load_model(model_path,compile=False)

    # Preprocess the image
    def preprocess_image(image_path, target_size=(128, 128)):
        image = cv2.imread(image_path)  # Load the image
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, target_size)  # Resize to target size
        image = preprocess_input(image)  # Normalize the image
        return np.expand_dims(image, axis=0)  # Add batch dimension

    # Preprocess and predict
    input_image = preprocess_image(image_path)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]

    # Prepare the output
    result = {
        "result": predicted_class_label,
        "class_probabilities": predictions.tolist()  # Convert to list for JSON compatibility
    }

    return result



@app.route('/tumor', methods=['POST','GET'])
def tumor():
	if request.method=="GET":
		return render_template('bonetumor.html')
	else:
		file = request.files["file"]
		

		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath,'uploads',  secure_filename(file.filename))
		file.save(file_path)
		prediction = predict_image_class(file_path,model_path='trained_modelvgg.h5')

		return render_template('disease-prediction-result.html', image_file_name=file.filename, result=prediction['result'])

@app.route('/uploads/<filename>')
def send_file(filename):
	return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
	app.run(debug=True)