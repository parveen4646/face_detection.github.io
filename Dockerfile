FROM python:3.9
WORKDIR /app



COPY . /app
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/keyfile.json"
#ENV GOOGLE_APPLICATION_CREDENTIALS="/app/keyfile.json"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

RUN apt-get update && apt-get install -y libglib2.0-dev libgl1-mesa-glx

CMD ["gunicorn", "-b", "0.0.0.0:8000", "main:app"]