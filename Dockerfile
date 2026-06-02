FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKDIR=/work

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /work \
    && chown -R app:app /app /work

COPY convert.py /app/convert.py

VOLUME ["/work"]
WORKDIR /work

USER app

ENTRYPOINT ["python", "/app/convert.py"]
