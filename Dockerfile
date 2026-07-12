FROM pennylaneai/pennylane:latest-lightning-gpu

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expects the dataset mounted at /workspace/data (see docker-compose.yml)
CMD ["python", "scripts/train.py", "data/nprint.csv", "model_out", "--backend", "lightning.gpu"]
