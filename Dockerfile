FROM --platform=linux/amd64 python:3.12-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install uv
RUN pip install --no-cache-dir uv==0.1.25

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy the application code
COPY ai_code_detector/ /app/ai_code_detector/
COPY models/ /app/models/
# Install dependencies using uv with system flag
RUN uv pip install --system -e . && \
    pip cache purge



# Create a non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser


# Expose the port the app runs on
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Command to run the application
CMD ["uvicorn", "ai_code_detector.api.app:app", "--host", "0.0.0.0", "--port", "8080"] 