# Second Life AI — container for Hugging Face Spaces (free CPU tier).
#
# Build context is the assembled Space repo produced by push.sh, which mirrors
# the source layout so app.py needs no path edits: it adds ../model to sys.path
# and finds its checkpoint under website/checkpoints/.

FROM python:3.11-slim

# Spaces runs containers as uid 1000 and routes traffic to $PORT (7860 by
# convention). HOST=0.0.0.0 is what app.py binds to; locally it stays loopback.
ENV PORT=7860 \
    HOST=0.0.0.0 \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd -m -u 1000 user
USER user
WORKDIR /app
ENV PATH="/home/user/.local/bin:$PATH"

# Dependencies first so image layers cache across content-only redeploys.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# The site, and the training repo's inference code it imports.
COPY --chown=user website/ website/
COPY --chown=user model/src/ model/src/

EXPOSE 7860

# --device cpu: the free tier has no GPU. ResNet-50 at 224px is ~0.3s/image
# there, and accuracy mode (4-view TTA) about four times that.
CMD ["python", "website/app.py", "--device", "cpu"]
