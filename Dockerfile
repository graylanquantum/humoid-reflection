# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    NLTK_DATA=/app/nltk_data

ARG TORCH_CHANNEL=cpu

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential python3-dev python3-tk \
    libgl1-mesa-glx libglib2.0-0 \
    cmake ninja-build git \
    curl iptables dnsutils openssl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash appuser && \
    mkdir -p /data /app && chown -R appuser:appuser /data /app

WORKDIR /app
COPY requirements.txt .
RUN sed -i '/^python-oqs\b/d' requirements.txt || true
RUN python -m pip install --upgrade pip wheel setuptools

# PyTorch install
RUN if [ "$TORCH_CHANNEL" = "cpu" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio ; \
    elif [ "$TORCH_CHANNEL" = "cu121" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio ; \
    else \
      echo "Unknown TORCH_CHANNEL='$TORCH_CHANNEL'" && exit 1 ; \
    fi

RUN pip install --no-cache-dir -r requirements.txt

# Pre-load NLTK data (-q quiet)
RUN mkdir -p /app/nltk_data && \
    python - << 'PY' || true
import nltk, os
NLTK_PATHS = [os.environ.get("NLTK_DATA","/app/nltk_data")]
for pkg in ["punkt", "averaged_perceptron_tagger", "wordnet", "stopwords", "conll2000"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_PATHS[0])
PY

# liboqs build
RUN git clone --depth=1 https://github.com/open-quantum-safe/liboqs /tmp/liboqs && \
    cmake -S /tmp/liboqs -B /tmp/liboqs/build -DOQS_USE_OPENSSL=OFF -GNinja && \
    cmake --build /tmp/liboqs/build && \
    cmake --install /tmp/liboqs/build && ldconfig && \
    rm -rf /tmp/liboqs

RUN pip install --no-cache-dir "git+https://github.com/open-quantum-safe/liboqs-python@0.12.0"

COPY . .
RUN chown -R appuser:appuser /app

# Default config.json
RUN printf '%s\n' '{\n  "DB_NAME":"dyson.db","API_KEY":"","WEAVIATE_ENDPOINT":"http://127.0.0.1:8079","WEAVIATE_QUERY_PATH":"/v1/graphql","MAX_TOKENS":2500,"CHUNK_SIZE":358\n}' > /app/config.json && \
    chown appuser:appuser /app/config.json

# Vault passphrase
RUN openssl rand -hex 32 > /app/.vault_pass && \
    echo "export VAULT_PASSPHRASE=$(cat /app/.vault_pass)" > /app/set_env.sh && \
    chmod +x /app/set_env.sh && chown appuser:appuser /app/.vault_pass /app/set_env.sh

# Firewall + model downloads
RUN cat << 'EOF' > /app/firewall_start.sh
#!/bin/bash
set -e
source /app/set_env.sh || true

iptables -F OUTPUT || true
iptables -A OUTPUT -o lo -j ACCEPT || true
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT || true
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT || true
iptables -A OUTPUT -j ACCEPT || true

BASE_MODEL_PATH=/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
BASE_MODEL_URL=https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
BASE_MODEL_SHA256=86c8ea6c8b755687d0b723176fcd0b2411ef80533d23e2a5030f845d13ab2db7

MM_PROJ_PATH=/data/llama-3-vision-alpha-mmproj-f16.gguf
MM_PROJ_URL=https://huggingface.co/abetlen/llama-3-vision-alpha-gguf/resolve/main/llama-3-vision-alpha-mmproj-f16.gguf
MM_PROJ_SHA256=ac65d3aeba3a668b3998b6e6264deee542c2c875e6fd0d3b0fb7934a6df03483

OSS_DIR=/data/gpt-oss-20b
mkdir -p "$OSS_DIR"

download_and_verify() {
  url="$1"; path="$2"; sha="$3"
  mkdir -p "$(dirname "$path")"
  if [ ! -f "$path" ]; then
    echo "Downloading: $url"
    curl -L --fail -o "$path" "$url"
  fi
  echo "$sha  $path" | sha256sum -c - || { echo "Checksum failed: $path"; exit 1; }
}

download_and_verify "$BASE_MODEL_URL" "$BASE_MODEL_PATH" "$BASE_MODEL_SHA256"
download_and_verify "$MM_PROJ_URL" "$MM_PROJ_PATH" "$MM_PROJ_SHA256"

download_and_verify "https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00000-of-00002.safetensors" "$OSS_DIR/model-00000-of-00002.safetensors" "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427"
download_and_verify "https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00001-of-00002.safetensors" "$OSS_DIR/model-00001-of-00002.safetensors" "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea"
download_and_verify "https://huggingface.co/openai/gpt-oss-20b/resolve/main/model-00002-of-00002.safetensors" "$OSS_DIR/model-00002-of-00002.safetensors" "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303"

# Tokenizer files
download_and_verify "https://huggingface.co/openai/gpt-oss-20b/resolve/main/tokenizer.json" "$OSS_DIR/tokenizer.json" "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3"
download_and_verify "https://huggingface.co/openai/gpt-oss-20b/resolve/main/tokenizer_config.json" "$OSS_DIR/tokenizer_config.json" "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070"
download_and_verify "https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json" "$OSS_DIR/config.json" "3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce"
download_and_verify "https://huggingface.co/openai/gpt-oss-20b/resolve/main/special_tokens_map.json" "$OSS_DIR/special_tokens_map.json" "dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293"

iptables -F OUTPUT || true
iptables -A OUTPUT -o lo -j ACCEPT || true
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT || true
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT || true

ALLOWED_DOMAINS="api.open-meteo.com api.coingecko.com huggingface.co objects.githubusercontent.com"
for d in $ALLOWED_DOMAINS; do
  getent ahosts "$d" | awk '/STREAM/ {print $1}' | sort -u | while read ip; do
    [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && iptables -A OUTPUT -d "$ip" -j ACCEPT || true
  done
done

iptables -A OUTPUT -j REJECT || true

exec su appuser -c "export DISPLAY=:0 NLTK_DATA=/app/nltk_data && python main.py"
EOF

RUN chmod +x /app/firewall_start.sh
CMD ["/app/firewall_start.sh"]
