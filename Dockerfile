# Step 1: Use an official lightweight Python runtime environment
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only requirements first to leverage Docker's caching mechanism
COPY requirements.txt .

# Step 4: Install dependencies inside the virtual system
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the remaining project folders into the container
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY config.py .

# Step 6: Expose port 8000 for network access
EXPOSE 8000

# Step 7: Run the FastAPI app using Uvicorn when the container starts
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]