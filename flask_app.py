

from flask import Flask, jsonify, render_template, redirect, url_for, request
from face import import_files, extract_face, generate_embedding, stack_embed, fina_result
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('process_faces.html')

@app.route('/process_faces')
def process_faces():
    folder_path = './images'
    list_of_images = import_files(folder_path)
    images_dict = {i: file for i, file in enumerate(list_of_images)}


    for image_path in list_of_images:
        faces = extract_face(image_path)
        embeddings, list_names = generate_embedding(faces)
        embeddd = stack_embed(embeddings)
        final_embeddings = embeddd
        fina_result(final_embeddings, list_names)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
