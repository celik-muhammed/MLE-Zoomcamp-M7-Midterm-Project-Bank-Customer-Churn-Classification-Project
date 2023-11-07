
# Use image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock into the container
# Copy the application code into the container
COPY ["churn-env", "model", "pycode", "./"]

# Install pipenv and the project dependencies
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# Expose the port that Flask will run on
EXPOSE 9696

# Start the Flask application using pipenv
# ENTRYPOINT ["python", "server_waitress.py"]
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "server_flask_app:app"]
