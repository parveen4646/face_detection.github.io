FROM python:3.9-slim
WORKDIR /app

COPY . /app
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/keyfile.json"
#ENV GOOGLE_APPLICATION_CREDENTIALS="/app/keyfile.json"

RUN python -m pip install -r requirements.txt

RUN apt-get update && apt-get install -y libglib2.0-dev libgl1-mesa-glx

CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]