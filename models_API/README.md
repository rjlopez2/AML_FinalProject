# Map Tile Colorization Application

This guide will help you set up and use the Map Tile Colorization application using Docker.

## 1. Installing Docker

### For Windows and macOS

1. **Download Docker Desktop:**
   - Go to the [Docker Desktop website](https://www.docker.com/products/docker-desktop).
   - Download the Docker Desktop installer for your operating system.

2. **Install Docker Desktop:**
   - Run the installer you downloaded and follow the installation instructions.

3. **Start Docker Desktop:**
   - After installation, open Docker Desktop from your applications list. Docker will start automatically and you should see the Docker icon in your system tray.

### For Linux

<details>
<summary>Click to expand</summary>

1. **Update your package index:**
   ```bash
   sudo apt-get update
   ```

2. **Install required packages:**
   ```bash
   sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
   ```s
s
3. **Add Docker’s official GPG key:**
   ```bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   ```

4. **Add the Docker APT repository:**
   ```bash
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   ```

5. **Install Docker:**
   ```bash
   sudo apt-get update
   sudo apt-get install docker-ce
   ```

6. **Verify Docker installation:**
   ```bash
   sudo systemctl status docker
   ```

   You should see that Docker is active and running.

</details>

## 2. Building the Docker Image

1. **Navigate to your project directory:**
   Open a terminal and change to the directory where your `Dockerfile` is located:
   ```bash
   cd AML_FinalProject/models_API
   ```

2. **Build the Docker image:**
   Use the following command to build your Docker image:
   ```bash
   docker build -t colorization .
   ```

   This command will read the `Dockerfile` and build the image according to the instructions provided.

## 3. Running the Application

1. **Run the Docker container:**
   After building the image, you can run the application using the following command:
   ```bash
   docker run --name colorization -d -p 8000:8000 colorization
   ```

   This command will start the container in detached mode (`-d`) and map port 80 inside the container to port 8000 on your host machine.

2. **Verify the container is running:**
   You can check if the container is running with:
   ```bash
   docker ps
   ```

   You should see your container listed with its ID and the port mappings.

## 4. Using the Application
NOTE: this is a relative heavy application. It may take couple of second antil is ready to be used.
1. **Open your web browser:**
   - Navigate to `http://localhost:8000`.

2. **Fill out the information:**
   - **Coordinates:** Enter the coordinates in the format `longitude,latitude;longitude,latitude`.
   - **Zoom Level:** Enter the zoom level.
   - **Layer Name:** no sure what to do with this. remove it?
   - **Times:** currently not working.
   - **Model:** Choose a model from the dropdown menu.

3. **Colorize:**
   - Click the "Colorize" button to send your data.
   - Have fun!

4. **View predictions:**
   - After submission, the page will display the original **color image**, the original **grey image**, and the **predicted image** side by side for each set of coordinates.

## Troubleshooting

- **Docker Not Running:** Ensure Docker is started and running. You should see the Docker icon in your system tray.
- **Port Conflicts:** Ensure port 8000 is not used by another application on your host machine. You can change the port mapping in the `docker run` command if needed.
- **Build Errors:** Check the `Dockerfile` and your project files for errors. Ensure all required files are in the correct locations.

---

Feel free to adjust any details according to your specific setup or additional instructions.