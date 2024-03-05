from flask import Flask, jsonify,render_template,redirect,url_for
from face import import_files, extract_face, generate_embedding, stack_embed, fina_result
import torch
app = Flask(__name__)
from facenet_pytorch import InceptionResnetV1
# Define your routes
@app.route('/')
def index():
    return render_template('process_faces.html')
@app.route('/process_faces')
def process_faces():
    list_of_images=import_files()
    images_dict={i:file for i,file in enumerate(list_of_images)}
    global model

    model=InceptionResnetV1(pretrained='vggface2').eval()
    imc=list(map(extract_face,list_of_images))
    imc=imc[0]
    embeddings,list_names=generate_embedding(imc,model)
    embeddd=stack_embed(embeddings)
    final_embeddings=embeddd
    fina_result(final_embeddings,list_names)
    return redirect(url_for('index'))

    



# Run the application
if __name__ == '__main__':
    app.run(debug=True)
