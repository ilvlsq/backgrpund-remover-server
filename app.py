from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io
import base64
from rembg import remove

app = Flask(__name__)
CORS(app)


def image_to_base64(image):
    img_io = io.BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    return base64.b64encode(img_io.getvalue()).decode()

@app.route('/filter', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    image = Image.open(file.stream)
    
    processed_images = []

    filters = [
        ('FIND_EDGES', ImageFilter.FIND_EDGES),
        ('CONTOUR', ImageFilter.CONTOUR),
        ('EMBOSS', ImageFilter.EMBOSS),
        ('BLUR', ImageFilter.BLUR),
        ('DETAIL', ImageFilter.DETAIL),
    ]
    
    processed_images.append(image_to_base64(image))
    
    for name, filter in filters:
        filtered_image = image.filter(filter)
        processed_images.append(image_to_base64(filtered_image))
    
    inverted_image = ImageOps.invert(image.convert('RGB'))
    processed_images.append(image_to_base64(inverted_image))
    
    grayscale_image = ImageOps.grayscale(image)
    processed_images.append(image_to_base64(grayscale_image))
    
    mirrored_image = ImageOps.mirror(image)
    processed_images.append(image_to_base64(mirrored_image))
    
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(1.5)
    processed_images.append(image_to_base64(bright_image))
    
    return jsonify({'images': processed_images})

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    image = Image.open(file.stream)
    output_image = remove(image)
    
    img_io = io.BytesIO()
    output_image.save(img_io, format='PNG')
    img_io.seek(0)
    
    base64_img = base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    return jsonify({'images': [base64_img]})

if __name__ == '__main__':
    app.run(debug=True)
