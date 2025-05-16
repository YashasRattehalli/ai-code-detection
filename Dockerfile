FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv==0.1.25

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy the application code
COPY ai_code_detector/ /app/ai_code_detector/
COPY models/ /app/models/
# Install dependencies using uv with system flag
RUN uv pip install --system -e . && \
    uv pip install --system torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu



# Create a non-root user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set Python path to include the app directory
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "ai_code_detector.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 