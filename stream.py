import os
import streamlit as st
import zipfile
import shutil
import tempfile
from google.cloud import storage
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import pandas as pd
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN


storage_client = storage.Client()
bucket_name = 'clusterimages'


def import_files(folder_path):
    list_of_images = os.listdir(folder_path)
    list_of_images = [filename for filename in list_of_images if filename.endswith('.jpeg')]
    return list_of_images


def extract_face(filename, required_size=(224, 224), extracted_images_dic1={},extracted_images_dir='/tmp/extracted_images' ):
    filename = os.path.join(extracted_images_dir, filename)
    if filename.endswith((".jpg", ".png", ".jpeg")):
        pixels = plt.imread(filename)
        mtcnn = MTCNN()
        results = mtcnn.detect_faces(pixels)
        extracted_images = []
        for i in range(0, len(results)):
            x1, y1, width, height = results[i]['box']
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = cv2.resize(face, required_size)
            extracted_images.append(image)
            extracted_images_dic = {filename: extracted_images}
            extracted_images_dic1.update(extracted_images_dic)
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
                plt.imsave(os.path.join(subdir, result_filename), imagee)

    files_in_directory = os.listdir(results_dir)
    if not files_in_directory:
        return 'Error: No files found in the result directory', 404

    zip_filename = 'result_images.zip'
    zip_path = os.path.join(results_dir, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, results_dir)
                zip_file.write(file_path, arcname=arcname)

    if os.path.exists(zip_path):
        return zip_path
    else:
        return f'Error: Zip file not found at {zip_path}', 404


def upload_file(file, filename):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_file(file)


def process_images(tmp_dir):
    results_dir = tempfile.mkdtemp()
    for root, dirs, files in os.walk(tmp_dir):
        if os.path.basename(root) == "__MACOSX":
            continue  # Skip processing files in __MACOSX directory
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                faces = extract_faces(image_path)
                embeddings = generate_embedding(faces)
                # Save results, create a ZIP file, etc.
    return results_dir

def main():
    st.title('Face Detection Application')

    uploaded_file = st.file_uploader("Upload a ZIP file", type=["zip"])
    if uploaded_file is not None:
        with st.spinner('Extracting files...'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    os.makedirs(tmp_dir, exist_ok=True)  # Create the temporary directory if it doesn't exist
                    zip_file_path = os.path.join(tmp_dir, uploaded_file.name)
                    with open(zip_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(tmp_dir)

                    results = process_images(tmp_dir)
                    st.success('Face detection completed!')
                    st.markdown(f"Download your results: [Download Zip File]({results})")
                except Exception as e:
                    st.error(f"Error processing images: {e}")

if __name__ == "__main__":
    main()