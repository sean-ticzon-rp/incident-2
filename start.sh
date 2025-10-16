#!/bin/bash
set -e

# Log the PORT being used
echo "🚀 Starting application on PORT: ${PORT:-8000}"
echo "📍 Binding to 0.0.0.0:${PORT:-8000}"

# Start uvicorn with Railway's PORT
exec uvicorn simple_api_upgraded:app \
  --host 0.0.0.0 \
  --port ${PORT:-8000} \
  --proxy-headers \
  --forwarded-allow-ips='*' \
  --log-level info