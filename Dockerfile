FROM python:3.9-slim
WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libglib2.0-dev libgl1-mesa-glx libgl1-mesa-dev



RUN python -m pip install -r requirements.txt


CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]

