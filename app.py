from flask import Flask, render_template, request
import pytesseract
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    text = ""
    if request.method == 'POST' and 'image' in request.files:
        image = Image.open(request.files['image'])
        text = pytesseract.image_to_string(image)
    return render_template('index.html', extracted_text=text)

if __name__ == '__main__':
    app.run(debug=True)
