FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# OS deps
RUN apt-get update && apt-get install -y \
      git ffmpeg libsndfile1-dev sox libsox-dev \
   && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Code
WORKDIR /app
COPY . .

ENV PYTHONPATH=/app
ENTRYPOINT ["asaca"]
CMD ["--help"]

