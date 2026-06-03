FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKDIR=/work

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Copy project metadata and sources into image for installation
COPY pyproject.toml README.md LICENSE requirements.txt /tmp/project/
COPY src /tmp/project/src

# Install declared dependencies (numpy) and the package itself
RUN pip install --no-cache-dir -r /tmp/project/requirements.txt \
    && pip install --no-cache-dir /tmp/project \
    && rm -rf /tmp/project

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /work \
    && chown -R app:app /app /work

VOLUME ["/work"]
WORKDIR /work

USER app

# Invoke the package as the canonical entrypoint
ENTRYPOINT ["python", "-m", "autoai_musical_play_video_chapters"]
