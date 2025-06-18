# Blur Project

## Prerequisites Commands

To set up the environment for the **Blur** project, follow these detailed steps:

### 1. Create a Virtual Machine
Set up a virtual machine using your preferred tool (e.g., VirtualBox, VMware, or a cloud provider). Ensure the virtual machine has Python installed.
For example :
```bash
python3 -m venv .venv
```
Then, enter in your virtual machine :
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
Run the following commands to install the required dependencies:

```bash
pip install -r requirements.txt
```
If you are on MacOS, make sure to install ffmpeg :
```bash
brew install ffmpeg
```

### 3. Prepare Files
Place your photos and videos in the `pics` folder. Photos can at least be in `png` or `jpg` format. Videos can at least be in `mp4` or `mov` format.

Now let's see what different options we have.

## Execution
We have 3 different cases :
### 1. Picture blur
Using the `blur_image` file, you can blur the faces clearly visible.

To use it, run the following command :
```bash
python blur_image.py name_of_picture.extension
```
If you want the pixelated version, add ```--pixel``` at the end of your command line.

### 2. Video blur
Same process here, except for the file name :
```bash
python blur_video.py name_of_video.extension
```
If you want the pixelated version, add ```--pixel``` at the end of your command line.

### 3. Real time video blur
You can also blur your face in real time with your camera :
```bash
python realtime_blur.py
```
You can switch between basic and pixelated blur by clicking on `b` or `p` (be sure to control the camera window with your keyboard, by clicking on it anywhere when it appears).


For every case, you can change the power of the pixelated blur by changing the `pixel_size` variable in each file (the higher it is, the less we will guess who is behind the blur).
