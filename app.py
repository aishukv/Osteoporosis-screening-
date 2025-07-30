import os
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
model = YOLO("best.pt")  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join("temp", filename)
        os.makedirs("temp", exist_ok=True)
        file.save(file_path)

        results = model.predict(source=file_path, save=False, conf=0.25)
        im_array = results[0].plot()

        # Convert image to base64 for display
        im_pil = Image.fromarray(im_array[..., ::-1])
        buf = io.BytesIO()
        im_pil.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        base64_image = base64.b64encode(byte_im).decode('utf-8')

        
        prediction = "Not detected"
        confidence = 0.0
        for box in results[0].boxes:
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            prediction = model.names[cls_id]
            confidence = round(conf * 100, 2)

        return render_template(
            'index.html',
            result_img=base64_image,
            prediction=prediction,
            confidence=confidence
        )
    return redirect(url_for('index'))

@app.route('/download-image', methods=['POST'])
def download_image():
    base64_data = request.form['image_data'].split(',')[1]
    image_bytes = base64.b64decode(base64_data)
    return send_file(
        io.BytesIO(image_bytes),
        mimetype='image/jpeg',
        as_attachment=True,
        download_name='detected_image.jpg'
    )

if __name__ == "__main__":
    app.run(debug=True)






