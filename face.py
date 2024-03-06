import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import pickle
import mtcnn
# Removing import of keras, as it's not used in the provided code

def import_files(folder_path):
    list_of_images=os.listdir(folder_path)
    #full_paths = [os.path.join(folder_path, filename) for filename in list_of_images]
    # Remove unwanted prefixes from the file paths

    print(os.listdir(os.getcwd()))
    print('files imported')
    print(len(list_of_images))
    return list_of_images


def extract_face(filename, required_size=(224, 224), extracted_images_dic1={}):
    filename_ = './images/'+ filename
    if filename_.endswith((".jpg", ".png",".jpeg")):
        # load image from file
        pixels = plt.imread(filename_)
        # create the detector, using default weights
        with open('mtcnn_detector.pkl', 'rb') as f:
            detector = pickle.load(f)
        # detect faces in the image
        results = detector.detect_faces(pixels)
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
    return extracted_images_dic1
    
        

def generate_embedding(pixels):
    with open('model.pkl','rb') as f:
        model=pickle.load(f)
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
        indices = np.where(score > 0.8)[0]  # np.where returns a tuple, so we access the first element ([0])
        result_indices.append(indices)
    xx = pd.Series([list(i) for i in result_indices if len(i) > 1]).drop_duplicates().reset_index(drop=True)
    print(xx)
    for m, i in enumerate(xx):
        
        for j in i:
            os.makedirs('images/'+str(m), exist_ok=True)
            k = list_names[j]
            imagee = plt.imread(f'./images/{k}')
            plt.imsave(f'./images/{str(m)}/{k}', imagee)
       
            print(f'its ={m}')
            print('result save')



