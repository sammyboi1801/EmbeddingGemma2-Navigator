---
title: EG2 Navigator
emoji: 🧭
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Text generation with a frozen EmbeddingGemma 2 encoder
---

# EG2 Navigator: model server

Streaming backend for the EG2 Navigator demo. A frozen EmbeddingGemma 2 encoder reads the
conversation and a 10.5M-parameter navigator writes the reply.

- `POST /chat`: streams one server-sent event per generated token.
- `GET /health`: readiness check.
- `GET /docs`: interactive API documentation.
