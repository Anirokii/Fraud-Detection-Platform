FROM python:3.10-slim

WORKDIR /app

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ajouter prometheus-client si pas dans requirements
RUN pip install prometheus-client==0.19.0

# Copier le code
COPY src/ ./src/
COPY models/ ./models/

# Exposer le port
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "src.api.fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]