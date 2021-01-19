# tello-magic

This program uses Yolov5 (PyTorch) and PicoVoice, enabling the Tello Drone to detect objects/humans and listen to voice commands via microphone. By analyzing the video stream coming from the drone, the program makes the decisions and sends commands back to the the drone. **This project is highly inspired by another great project using Openpose to control the Tello drone (https://github.com/geaxgx/tello-openpose), which I took as a reference.** 
Also a big thanks to everyone behind https://github.com/hanyazou/TelloPy, which made the progress way more fun.

<img src="/HUD.jpg" alt="HUD" style="zoom: 80%;" />

This is a private project of mine testing deep learning in combination with drones. It's still a work in progress, so don't expect a final product. 

##### The program was tested on following setup:

- Windows 10 Pro (Build 19042)
- AMD Ryzen 7 1800X (3600 Mhz, 8 Core)
- Nvidia GTX 980 (4GB Video Ram)
- 32 GB Memory
- Python 3.8.5

### Required Packages:

#### ***Important:*** 

Make sure you have CUDA/CUDNN installed and running to use PyTorch with GPU support. A Nvidia GTX 980 or better is recommended, otherwise the fps will drop and the drone PID will not work correctly or has to be retuned.

#### Base

- Python> 3.8.5
- opencv-python>=4.4.0.46
- pynput>=1.7.1
- TelloPy>= 0.6.0 https://github.com/hanyazou/TelloPy
- PyAV>= 8.0.2  https://github.com/PyAV-Org/PyAV
- simple-pid >= 0.2.4  https://github.com/m-lundberg/simple-pid

#### Yolov5

- Cython
- matplotlib>=3.2.2
- numpy>=1.18.5
- Pillow
- PyYAML>=5.3
- scipy>=1.4.1
- tensorboard>=2.2
- torch>=1.7.0
- torchvision>=0.8.
- tqdm>=4.41.0

#### PicoVoice

- PyAudio>= 0.2.11 https://people.csail.mit.edu/hubert/pyaudio/
- pvrhinodemo>= 1.6.0 https://picovoice.ai/docs/quick-start/rhino-python/



### Usage:

The drone can be controlled via keyboard (for the **detailed keyboard setup** have a look in the controller.py under init_controlls) and can take several voice commands (see **voice_commands.txt**). Make sure there is **enough light** for the drones floor sensor and your microphone level is adjusted. 

**The PicoVoice voice model was created on 01-18-21 and is only valid for 1 month, since it is a private account for testing. If the model is out of date or you want to modify the voice commands, visit https://picovoice.ai/, create a personal account and create your individual voice recognition model with Rhino. You have to modify the process_audio function in tello_magic.py then.**

Before you start: With **"ESC"** you can always abort and land immediately. With **"T"** the drone will activate/deactivate all tracking modes and stop moving.

#### Quickstart:

1. Unrar yolo5/weights/tello_custom.pt (tello_custom.part1 & part2)

2. If your setup is correct, all packages installed, microphone (Bluetooth headphone also works) is plugged in and you are connected to your tello via wifi, open a terminal and start the main process (takes ~ 15 sec to start up): 

   --> python tello_magic.py

3. If there are no errors you should see the drones video feed with HUD information in a separate window after a little while

4. You can start the drone with [TAB] on the keyboard or by saying "takeoff"

5. You can control the drone with Q,W,E,A,S,D and the arrow keys. To land the drone, press ESC or say "land".

6. If you say "follow me", the drone will track your head and by saying "lock distance", the drone always tries to keep the current distance 

7. You can even say "take a picture of a guitar" and the drone will autonomously search for a guitar, approach it and take a picture

   This is currently limited to 6 objects (person, hand, head, orange, guitar, chair) .

8. In distance tracking mode, you can raise your hand to the height of your head to move the drone to the left or right

9. Say "land" or "palm land" to land the drone on the floor or on your hand

Note: For further commands and actions look at the voice_commands.txt file

### Details:

**tello_magic.py**

This is the main script to start all processes in parallel (Multiprocessing). I also initializes the necessary shared memory blocks to share data between the processes on the machine. The advantage of splitting into multiple processes such as the main drone control, yolo object detection, voice recognition or video decoding is a significant increase of calculated frames per second. 


**yolov5/detect.py**

This is the Yolo Object Detection Class. The detect function returns the current frame with added bounding boxes and updates a shared numpy array with information about every object found in the frame.


**rhino_get_audio.py**

This Class is responsible for the voice recognition and the run function updates a shared numpy array with current voice command.


**FPS.py**

Defines class FPS, which calculates and displays the current frames/second.

**CameraMorse.py**

This is originally from the tello-openpose project and has some useful functions such as the RollingGraph to finetune the drones PID.


**simple-pid:**

Used here to control the yaw, pitch, rolling and throttle of the drone.

The parameters of the PIDs may depend on the processing speed and need tuning to adapt to the FPS you can get. For instance, if the PID that controls the yaw works well at 20 frames/sec, it may yield oscillating yaw at 10 frames/sec.



### Potential Addons for the Future:

- Adding Visual Odometry for monocular camera systems like OrbSlam2 for 3D positioning or even collision detection 
- Adding Realtime Audio Noise Filter to reduce drone noise in mic
- Adding more Voice Commands

