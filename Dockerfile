FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ffmpeg is required by convert.py when extracting audio from input media.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Runtime Python dependency for audio/block computations.
RUN pip install --no-cache-dir numpy

WORKDIR /work

# Keep defaults overridable via -e flags.
ENV CHUNK_CHARS=12000

CMD ["python", "/work/convert.py"]
