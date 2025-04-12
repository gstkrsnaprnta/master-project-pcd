from flask import Flask, request, render_template, jsonify, send_file
import os
from time import time
import logging
from effects.ghibli_effect import create_ghibli_effect
from effects.grayscale_effect import convert_to_grayscale
from effects.edge_effect import detect_edges
from effects.cartoon_effect import create_cartoon_effect
from effects.posterize_effect import posterize_image
from effects.binary_effect import binary_threshold
from effects.oil_painting_effect import oil_painting_effect
from effects.pencil_sketch_effect import pencil_sketch_effect
from effects.emboss_effect import emboss_effect
from effects.watercolor_effect import watercolor_effect
from effects.resample_effect import resample_image

# Setup logging untuk debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    conversion_type = request.form.get('conversion_type', 'ghibli')
    resolution = request.form.get('resolution', 'original')
    width = request.form.get('width')
    height = request.form.get('height')
    output_filename = f"{conversion_type}_{int(time())}_{file.filename}"
    temp_path = os.path.join(OUTPUT_FOLDER, f"temp_{output_filename}")
    final_output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    file.save(input_path)
    logging.debug(f"File saved to: {input_path}")
    
    try:
        if conversion_type == 'ghibli':
            result_path, warning = create_ghibli_effect(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Ghibli effect: {temp_path}")
            logging.debug(f"Ghibli effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
            if warning:
                response['warning'] = warning
        elif conversion_type == 'grayscale':
            result_path = convert_to_grayscale(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Grayscale effect: {temp_path}")
            logging.debug(f"Grayscale effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        elif conversion_type == 'edge':
            result_path = detect_edges(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Edge effect: {temp_path}")
            logging.debug(f"Edge effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        elif conversion_type == 'cartoon':
            result_path = create_cartoon_effect(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Cartoon effect: {temp_path}")
            logging.debug(f"Cartoon effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        elif conversion_type == 'posterize':
            result_path = posterize_image(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Posterize effect: {temp_path}")
            logging.debug(f"Posterize effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        elif conversion_type == 'binary':
            result_path = binary_threshold(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Binary effect: {temp_path}")
            logging.debug(f"Binary effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        elif conversion_type == 'oil_painting':
            result_path = oil_painting_effect(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Oil Painting effect: {temp_path}")
            logging.debug(f"Oil Painting effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        elif conversion_type == 'pencil_sketch':
            result_path = pencil_sketch_effect(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Pencil Sketch effect: {temp_path}")
            logging.debug(f"Pencil Sketch effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        elif conversion_type == 'emboss':
            result_path = emboss_effect(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Emboss effect: {temp_path}")
            logging.debug(f"Emboss effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        elif conversion_type == 'watercolor':
            result_path = watercolor_effect(input_path, temp_path)
            if not os.path.exists(temp_path):
                raise ValueError(f"Failed to create temp file for Watercolor effect: {temp_path}")
            logging.debug(f"Watercolor effect applied, temp file: {temp_path}")
            response = {
                'message': 'Image processed',
                'output_url': f'/static/outputs/{output_filename}',
                'download_url': f'/download?file={output_filename}'
            }
        else:
            raise ValueError(f"Unknown conversion type: {conversion_type}")

        # Resample ke resolusi yang dipilih
        resample_image(temp_path, final_output_path, resolution, width, height)
        logging.debug(f"Image resampled to: {final_output_path}")

        # Bersihkan file sementara
        os.remove(input_path)
        os.remove(temp_path)
        return jsonify(response)

    except ValueError as e:
        logging.error(f"ValueError: {str(e)}")
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/download', methods=['GET'])
def download_file():
    filename = request.args.get('file')
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    logging.debug(f"Downloading file: {file_path}")
    return send_file(file_path, as_attachment=True, download_name=filename)
    
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # Kirim file dengan header untuk unduhan
    return send_file(file_path, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    app.run(debug=True)