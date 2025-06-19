
FROM python:3.9-slim

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /pharmacoeconomics

COPY requirements.txt ./
COPY . .

RUN pip install -r requirements.txt