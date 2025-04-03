from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import time
import uuid
from emotion_analyzer import EmotionAnalyzer
from image_generator import ImageGenerator
from style_transfer import StyleTransfer

app = Flask(__name__)

# Initialize components
emotion_analyzer = EmotionAnalyzer()
image_generator = ImageGenerator()
style_transfer = StyleTransfer()

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Available styles
STYLES = {
    "none": "No Style",
    "starry_night": "Starry Night (Van Gogh)",
    "kandinsky": "Kandinsky Abstract",
    "picasso": "Picasso Cubism",
    "monet": "Monet Impressionism"
}

@app.route('/')
def index():
    return render_template('index.html', styles=STYLES)

@app.route('/generate', methods=['POST'])
def generate():
    # Get form data
    prompt = request.form.get('prompt', '')
    style_name = request.form.get('style', 'none')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Analyze emotion in the text
        emotion_data = emotion_analyzer.analyze_emotion(prompt)
        
        # Generate a unique ID for this generation
        generation_id = str(uuid.uuid4())
        
        # Generate the base image
        base_image = image_generator.generate_image(prompt, emotion_data)
        base_image_path = image_generator.save_image(base_image, f"{generation_id}_base.jpg")
        
        # Apply style transfer if a style is selected
        if style_name != 'none':
            final_image_path = os.path.join("static/images", f"{generation_id}_styled.jpg")
            style_transfer.apply_style(base_image_path, style_name, final_image_path)
            image_url = url_for('static', filename=f"images/{generation_id}_styled.jpg")
        else:
            image_url = url_for('static', filename=f"images/{generation_id}_base.jpg")
        
        # Return the result
        return jsonify({
            'success': True,
            'image_url': image_url,
            'emotion': emotion_data['dominant_emotion'],
            'emotion_score': emotion_data['emotion_score'],
            'prompt': prompt,
            'style': STYLES.get(style_name, 'None')
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_style', methods=['POST'])
def upload_style():
    if 'style_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['style_image']
    style_name = request.form.get('style_name', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not style_name:
        return jsonify({'error': 'No style name provided'}), 400
    
    try:
        # Save the uploaded style image
        filename = secure_filename(f"{style_name}.jpg")
        filepath = os.path.join(style_transfer.style_dir, filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Style "{style_name}" uploaded successfully',
            'style_name': style_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
