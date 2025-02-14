from flask import Flask, request, render_template
from ocr import extract_text
from rag_model import generate_answer  # oder generate_rag_answer, wenn interner Kontext genutzt wird

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    if request.method == 'POST':
        # Option 1: Datei-Upload für OCR
        if 'file' in request.files:
            file = request.files['file']
            image_path = "uploaded_image.png"
            file.save(image_path)
            extracted_text = extract_text(image_path)
            result = f"<b>Extrahierter Text:</b><br>{extracted_text}"
        
        # Option 2: Benutzerfrage eingeben
        elif 'question' in request.form:
            question = request.form['question']
            # Hier könnte auch die Funktion generate_rag_answer genutzt werden, falls interner Kontext verfügbar ist.
            result = generate_answer(question)
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
import atexit

def clean_exit():
    print("Server wird beendet, speichere Ressourcen...")
    # Hier kannst du offene Prozesse beenden oder Dateien schließen

atexit.register(clean_exit)
