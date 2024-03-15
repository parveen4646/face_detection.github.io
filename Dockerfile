FROM python:3.9-slim
WORKDIR /app

COPY . /app
COPY keyfile.json /app/
RUN apt-get update && apt-get install -y libglib2.0-dev libgl1-mesa-glx libgl1-mesa-dev

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/keyfile.json"
#ENV GOOGLE_APPLICATION_CREDENTIALS="/app/keyfile.json"
ENV EXTRACTED_IMAGES_DIR=/tmp/extracted_images
RUN python -m pip install -r requirements.txt


CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]

