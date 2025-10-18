FROM ollama/ollama:0.12.6

ARG OLLAMA_MODEL=llama3.1:8b

ENV OLLAMA_HOST=0.0.0.0 \
    OLLAMA_MODEL=${OLLAMA_MODEL}

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && ollama pull "${OLLAMA_MODEL}"

EXPOSE 11434

CMD ["serve"]
