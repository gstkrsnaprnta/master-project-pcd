from flask import Flask, request, send_file, render_template, jsonify
import os
from time import time
from ghibli_effect import create_ghibli_effect

app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    if not os.path.splitext(file.filename)[1].lower() in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Only JPG, JPEG, and PNG are allowed.'}), 400
    
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_filename = f"ghibli_{int(time())}_{file.filename}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    file.save(input_path)
    
    try:
        result_path, warning = create_ghibli_effect(input_path, output_path)
        os.remove(input_path)
        response = {
            'message': 'Image processed successfully',
            'output_url': f'/static/outputs/{output_filename}'
        }
        if warning:
            response['warning'] = warning
        return jsonify(response)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)