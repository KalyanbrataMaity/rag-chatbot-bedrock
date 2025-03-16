# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8085

# Command to run the Streamlit app
CMD [ "streamlit", "run", "app.py", "--server.port=8085", "--server.address=0.0.0.0" ]
