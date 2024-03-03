
from mtcnn import MTCNN
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
import torch
from facenet_pytorch import InceptionResnetV1



def import_files():
    list_of_images=os.listdir('/Users/Guchi/Documents/projects/face detection model/images')
    print('filesimported')
    print(len(list_of_images))
    return  list_of_images


def extract_face(filename, required_size=(224, 224), extracted_images_dic1={}):
    filename_='images/'+str(filename)
    # load image from file
    pixels = plt.imread(filename_)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    extracted_images=[]
    for i in range(0,len(results)):
        x1, y1, width, height = results[i]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image=cv2.resize(face,required_size)
        extracted_images.append(image)
        extracted_images_dic={filename:extracted_images}
        extracted_images_dic1.update(extracted_images_dic)
    return extracted_images_dic1




def generate_embedding(pixels):
    emb_list=[]
    name_list=[]
    for v,i in pixels.items():
        for sep in i:
            face=torch.tensor(sep).to(device)
            face=(face/255.0-0.5)/.5
            face=face.unsqueeze(0)
            tensor_permuted = face.permute(0, 3, 1, 2)
            embeddings=model(tensor_permuted)
            emb_list.append(embeddings)
            name_list.append(v)
    return emb_list,name_list






def stack_embed(embeddings):
    stacked_embeddings = torch.cat(embeddings, dim=0)
    return stacked_embeddings


from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 

def fina_result(final_embeddings,list_names):
    print('final loop')
    print(os.getcwd())
    result_indices = []
    print(final_embeddings.shape[0])
    for i in range(final_embeddings.shape[0]):
        score = torch.nn.functional.cosine_similarity(final_embeddings[i], final_embeddings)
        indices = np.where(score > 0.8)[0]  # np.where returns a tuple, so we access the first element ([0])
        result_indices.append(indices)
    xx=pd.Series([list(i) for i in result_indices if len(i)>1]).drop_duplicates().reset_index(drop=True)
    print(xx)
    for m,i in enumerate(xx):
        os.makedirs(str(m),exist_ok=True)
        for j in i:
            k=list_names[j]
            l=f'./images/{k}'
            imagee=plt.imread(l)
            plt.imsave(f'./{m}/{k}',imagee)
            print('result save')
    
    

def main():
    list_of_images=import_files()
    images_dict={i:file for i,file in enumerate(list_of_images)}
    global device
    global model
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=InceptionResnetV1(pretrained='vggface2').eval().to(device)
    imc=list(map(extract_face,list_of_images))
    imc=imc[0]
    embeddings,list_names=generate_embedding(imc)
    embeddd=stack_embed(embeddings)
    final_embeddings=embeddd
    fina_result(final_embeddings,list_names)


if __name__ == "__main__":
    main()

 

