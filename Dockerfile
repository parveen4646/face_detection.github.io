# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Expose port 5000 to the outside world
EXPOSE 8000

# Define the command to run the Flask application
CMD ["gunicorn", "-b", "0.0.0.0:8000", "flask_app:app"]
