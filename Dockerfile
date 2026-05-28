FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    cmake make build-essential \
    libopenblas-dev liblapack-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    wget bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download dlib model files directly
RUN wget -q "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" \
    && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

RUN wget -q "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2" \
    && bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "deployment_code:app", "--host", "0.0.0.0", "--port", "8000"]
