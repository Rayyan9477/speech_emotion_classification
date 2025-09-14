FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# System deps for librosa, soundfile, and plotly/kaleido
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    graphviz \
    libgomp1 \
    libgthread-2.0-0 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

# Allow switching between Streamlit UI and FastAPI service later via MODE env
ENV MODE=streamlit

CMD ["bash", "-lc", "if [ \"$MODE\" = \"api\" ]; then uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT}; else streamlit run src/ui/streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true; fi"]


