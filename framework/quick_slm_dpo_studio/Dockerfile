# DPO Studio, mock mode. Build from the repository ROOT so that both halves of
# the app are in the build context: the backend lives under framework/ and the
# UI under app/, and an image needs each.
#
#     docker build -f framework/quick_slm_dpo_studio/Dockerfile -t quick-slm-dpo .
#     docker run --rm -p 8000:8000 \
#       -v "$PWD/framework/quick_slm_dpo_studio/data:/srv/data" quick-slm-dpo
#
# or just `docker compose up` from framework/quick_slm_dpo_studio/.
#
# This image is deliberately lean: mock mode only needs numpy (a base dep of the
# trainer's import chain) plus the web server. torch and transformers are NOT
# installed. Teacher mode (Gemma 4) runs on the GPU box from a CUDA base image;
# see README.md, "Teacher mode".
FROM python:3.11-slim

WORKDIR /srv

RUN pip install --no-cache-dir numpy fastapi "uvicorn[standard]"

# Only what the app imports: the framework packages (the trainer for
# template/tools/prompts, and the studio backend itself) and the UI it serves.
# `framework` goes on the path rather than being pip-installed, so the mock image
# stays small and builds in seconds.
COPY framework/quick_slm_trainer ./framework/quick_slm_trainer
COPY framework/quick_slm_dpo_studio ./framework/quick_slm_dpo_studio
COPY app/quick_slm_dpo_studio ./app/quick_slm_dpo_studio

ENV PYTHONPATH=/srv/framework \
    PYTHONUNBUFFERED=1 \
    DPO_MODE=mock \
    DPO_DATA_PATH=/srv/data/preferences.jsonl \
    DPO_UI_DIR=/srv/app/quick_slm_dpo_studio

EXPOSE 8000
VOLUME ["/srv/data"]

CMD ["uvicorn", "quick_slm_dpo_studio.app:app", "--host", "0.0.0.0", "--port", "8000"]
