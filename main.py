from flask import Flask, render_template, request, send_file
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import zipfile
import shutil
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
import tempfile
import pandas as pd
import pickel


app = Flask(__name__)

# Define the directory to store extracted images
# Use the mounted directory in Kubernetes if available, otherwise fallback to a local directory
temp_dir = tempfile.mkdtemp()
extracted_images_dir=temp_dir


def import_files(folder_path):
    list_of_images = os.listdir(folder_path)
    list_of_images = [filename for filename in list_of_images if filename.endswith('.jpeg')]
    return list_of_images


def extract_face(filename, required_size=(224, 224), extracted_images_dic1={}):
    global extracted_images_dir
    filename = os.path.join(extracted_images_dir, filename)
    if filename.endswith((".jpg", ".png", ".jpeg")):
        print(f'filename={filename}')
        # load image from file
        pixels = plt.imread(filename)
        # create the detector, using default weights
        mtcnn = MTCNN()
        results = mtcnn.detect_faces(pixels)
        # extract the bounding box from the first face
        extracted_images = []
        for i in range(0, len(results)):
            x1, y1, width, height = results[i]['box']
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = cv2.resize(face, required_size)
            extracted_images.append(image)
            extracted_images_dic = {filename: extracted_images}
            extracted_images_dic1.update(extracted_images_dic)
    else:
        print(f'else {filename}')
    return extracted_images_dic1


def generate_embedding(pixels):
    model = InceptionResnetV1(pretrained='vggface2').eval()
    emb_list = []
    name_list = []
    if pixels is not None:
        for v, i in pixels.items():
            for sep in i:
                face = torch.tensor(sep)
                face = (face / 255.0 - 0.5) / .5
                face = face.unsqueeze(0)
                tensor_permuted = face.permute(0, 3, 1, 2)
                embeddings = model(tensor_permuted)
                emb_list.append(embeddings)
                name_list.append(v)
    return emb_list, name_list


def stack_embed(embeddings):
    embeddings = [i for i in embeddings if i is not None and len(i) > 0]
    stacked_embeddings = torch.cat(embeddings, dim=0)
    return stacked_embeddings


def final_result(final_embeddings, list_names):
    global extracted_images_dir
    print('final loop')
    result_indices = []
    for i in range(final_embeddings.shape[0]):
        score = torch.nn.functional.cosine_similarity(final_embeddings[i], final_embeddings)
        indices = np.where(score > 0.8)[0]
        result_indices.append(indices)

    xx = pd.Series([list(i) for i in result_indices if len(i) > 1]).drop_duplicates().reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        results_dir = tmp_dir
        for m, i in enumerate(xx):
            for j in i:
                k = list_names[j]
                imagee = plt.imread(k)
              

                subdir = os.path.join(results_dir, str(m))
                os.makedirs(subdir, exist_ok=True)

                result_filename = f'result_{m}_face_{os.path.basename(k)}'
                print('result_filename=={result_filename}')
                plt.imsave(os.path.join(subdir, result_filename), imagee)

        print(f"Temporary directory contents: {os.listdir(results_dir)}")
        files_in_directory = os.listdir(results_dir)
        print(f"Files in result directory: {files_in_directory}")

        if not files_in_directory:
            return 'Error: No files found in the result directory', 404

        # Zip the result directory using the zipfile module
        zip_filename = 'result_images.zip'
        zip_path = os.path.join(results_dir, zip_filename)

        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, results_dir)
                    zip_file.write(file_path, arcname=arcname)

        # Debugging statements
        print(f"Zip file path: {zip_path}")
        print(f"Zip file exists: {os.path.exists(zip_path)}")
        print(f"Files added to the zip file: {files_in_directory}")

        # Check if the zip file exists
        if os.path.exists(zip_path):
            # Send the zip file as an attachment
            return send_file(zip_path, as_attachment=True), shutil.rmtree(results_dir)
        else:
            return f'Error: Zip file not found at {zip_path}', 404


@app.route('/')
def index():
    # Render the HTML form for folder upload
    return render_template('base.html')


@app.route('/upload', methods=['POST'])
def upload_folder():
    global extracted_images_dir
    if 'folder' not in request.files:
        return 'No folder part'

    folder = request.files['folder']
    if folder.filename == '':
        return 'No selected folder'
    print(f'folder.filename{folder.filename}')
    if folder.filename.endswith('.zip'):
        zip_file_path = os.path.join(extracted_images_dir, folder.filename)
        zip_file_path=str(zip_file_path).replace('/f1/','/')
        folder.save(zip_file_path)
        print(f'zip file path{zip_file_path}')
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_images_dir)

        os.remove(zip_file_path)

        list_of_images = []  # Initialize list_of_images here

        try:
            list_of_images = import_files(extracted_images_dir)
        except Exception as e:
            shutil.rmtree(extracted_images_dir)
            return f"Error occurred while importing files: {e}", 500

        extracted_images_dic1 = {}
        final_embeddings = None
        list_names = []

        for image_path in list_of_images:
            extracted_faces = extract_face(image_path, extracted_images_dic1=extracted_images_dic1)
            if extracted_faces:
                embeddings, temp_list_names = generate_embedding(extracted_faces)
                list_names.extend(temp_list_names)
                embeddd = stack_embed(embeddings)
                final_embeddings = embeddd

        if final_embeddings is not None:
            return final_result(final_embeddings, list_names)
        else:
            shutil.rmtree(extracted_images_dir)
            return 'Error: No faces extracted from the uploaded images', 404
    else:
        return "no zip folder{list_of_images}"

    shutil.rmtree(extracted_images_dir)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
