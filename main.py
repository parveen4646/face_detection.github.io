from flask import Flask, render_template, redirect, url_for, request,send_file
import os
from google.cloud import storage
app = Flask(__name__)
storage_client = storage.Client()
bucket_name = 'clusterimages'
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import zipfile
import shutil
import pickle
from facenet_pytorch import InceptionResnetV1
import tempfile
from mtcnn import MTCNN
import matplotlib
matplotlib.use('Agg')

# Removing import of keras, as it's not used in the provided code
extracted_images_dir = '/tmp/extracted_images'
def import_files(folder_path):
    
    list_of_images = os.listdir(folder_path)
    list_of_images = [filename for filename in list_of_images if filename.endswith('.jpeg')]

    print(f'currentcwd{os.getcwd()}')
    #full_paths = [os.path.join(folder_path, filename) for filename in list_of_images]
    # Remove unwanted prefixes from the file paths
    print('files imported')
    print(len(list_of_images))
    print(list_of_images)
    return list_of_images


def extract_face(filename, required_size=(224, 224), extracted_images_dic1={}):
    filename=os.path.join(extracted_images_dir,filename)
    filename_ = filename
    if filename_.endswith((".jpg", ".png",".jpeg")):
        print(f'filename={filename}')
        # load image from file
        pixels = plt.imread(filename_)
        # create the detector, using default weights
        mtcnn=MTCNN()
        results= mtcnn.detect_faces(pixels)
        # extract the bounding box from the first face
        extracted_images=[]
        for i in range(0, len(results)):
            x1, y1, width, height = results[i]['box']
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = cv2.resize(face, required_size)
            extracted_images.append(image)
            extracted_images_dic = {filename: extracted_images}
            extracted_images_dic1.update(extracted_images_dic)
    else:
        print(f'else{filename}')        
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


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd



def fina_result(final_embeddings, list_names):
    print('final loop')
    print(os.getcwd())
    result_indices = []
    print(final_embeddings.shape[0])
    for i in range(final_embeddings.shape[0]):
        score = torch.nn.functional.cosine_similarity(final_embeddings[i], final_embeddings)
        indices = np.where(score > 0.8)[0]
        result_indices.append(indices)

    xx = pd.Series([list(i) for i in result_indices if len(i) > 1]).drop_duplicates().reset_index(drop=True)
    print(xx)
    with tempfile.TemporaryDirectory() as tmp_dir:
        results_dir = tmp_dir
    for m, i in enumerate(xx):

        for j in i:
            k = list_names[j]
            imagee = plt.imread(k)

            subdir = os.path.join(results_dir, str(m))
            os.makedirs(subdir, exist_ok=True)

            result_filename = f'result_{m}_face_{os.path.basename(k)}'
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
        return send_file(zip_path, as_attachment=True)
    else:
        return f'Error: Zip file not found at {zip_path}', 404



def upload_file(file, filename):
    # Upload file to Cloud Storage
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_file(file)


@app.route('/')
def index():
    # Render the HTML form for folder upload
    return render_template('base.html')
    
@app.route('/upload', methods=['POST'])
def upload_folder():
    if 'folder' not in request.files:
        return 'No folder part'
    
    folder = request.files['folder']
    if folder.filename == '':
        return 'No selected folder'
    
    
    # Extract images from the uploaded zip folder using os.listdir
    extracted_images_dir = '/tmp/extracted_images'
    os.makedirs(extracted_images_dir, exist_ok=True)
    if folder.filename.endswith('.zip'):
        with zipfile.ZipFile(folder, 'r') as zip_ref:
            zip_ref.extractall(extracted_images_dir)

        # Import and process each image using import_files and extract_face functions
        list_of_images = import_files(extracted_images_dir)
        extracted_images_dic1 = {}

    
    for image_path in list_of_images:
        #image_path=os.join.path(os.path(extracted_images_dir),image_path)
        # Extract faces using the function
        extracted_faces = extract_face(image_path, extracted_images_dic1=extracted_images_dic1)
        if extracted_faces:
            embeddings, list_names = generate_embedding(extracted_faces)
            embeddd = stack_embed(embeddings)
            final_embeddings = embeddd
            
        # Perform further processing or upload the extracted faces as needed
        # For example, you can upload each face to Cloud Storage
     

    # Remove the extracted_images_dir after processing
    #shutil.rmtree(extracted_images_dir)

    success_message = 'Folder uploaded, images processed, and faces extracted successfully'
    return fina_result(final_embeddings, list_names)
    


if __name__ == '__main__':
    app.run(debug=True,port=8000)
