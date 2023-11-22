# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to C:\app
WORKDIR C:\app

# Install necessary system packages
# Note: Windows containers don't use apt-get, so we skip this step on Windows

# Copy the current directory contents into the container at C:\app
COPY . .

# Install TA-Lib
# COPY app/Tab-Lib-deps/ta-lib-0.4.0-src.tar.gz .
# RUN New-Item -ItemType Directory -Path C:\ta-lib
# RUN tar -zxvf .\ta-lib-0.4.0-src.tar.gz -C C:\ta-lib --strip-components=1
# RUN Remove-Item .\ta-lib-0.4.0-src.tar.gz -Force

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME trading-bot

# Run app.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]
