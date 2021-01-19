#This program uses Yolov5 (PyTorch) and PicoVoice, enabling the Tello Drone to detect objects/humans and listen to voice commands via microphone. 
#The project is highly inspired by another great project using Openpose to control the Tello drone (https://github.com/geaxgx/tello-openpose), which I took as a reference. 
#Also a big thanks to everyone behind https://github.com/hanyazou/TelloPy, which made the progress way more fun.


import sys
sys.path.insert(1, 'yolov5')
import time
import numpy as np
import cv2
import argparse
from detect import *
from multiprocessing import Process, shared_memory
from controller import *
import logging
import rhino_get_audio



def start_rhino():
    """Start Rhino Voice Recognition Process"""

    rhino_get_audio.main()
    print('Voice Listener startet')


def detect_objects():
    """
    Detect objects in frame(from sharred array) and return target information 
    """
    detector = yolo()

    detect_fps = FPS()

    shm_to_yolo = shared_memory.SharedMemory(name='img_to_yolo')
    shared_img_to_yolo = np.ndarray((720, 960, 3), dtype=np.uint8, buffer=shm_to_yolo.buf)

    # existing_shm_detect = shared_memory.SharedMemory(name='detection')
    # shared_detection_array = np.ndarray((4,), dtype=np.int32, buffer=existing_shm_detect.buf)

    existing_shm_detect_all = shared_memory.SharedMemory(name='detection_all')
    shared_detection_array_all = np.ndarray((6,4), dtype=np.int32, buffer=existing_shm_detect_all.buf)

    shm_from_yolo = shared_memory.SharedMemory(name='img_from_yolo')
    shared_img_from_yolo = np.ndarray((720, 960, 3), dtype=np.uint8, buffer=shm_from_yolo.buf)

    while True:

        frame, all_array = detector.detect(shared_img_to_yolo, show_img=False)
        #shared_detection_array[:] = [target_x, target_y, target_width, target_height]
        shared_img_from_yolo[:] = frame[:]
        shared_detection_array_all[:] = all_array[:]

        detect_fps.update()
        #print(f'detect_fps: {detect_fps.get()}')


def main(log_level=None):
    """Main Function to start Tello Processes"""

    #Initialize the Tello Controller
    tello = TelloController(use_face_tracking=True, 
                            kbd_layout="QWERTY", 
                            write_log_data=False, 
                            log_level=None)
    
    #Wait for a continous video stream
    time.sleep(10)
    tello.get_frames()
    time.sleep(10)
    first_frame = True  
    frame_skip = 300
    
    #Create Shared Array for Image share from main to yolo process
    shared_img_to_yolo = None

    #Create Shared Array for Image share from yolo to main process
    shared_img_from_yolo = None


    # #Create Shared Array for yolo detection results
    # detection_array = np.array([0,0,0,0])
    # shm_detect = shared_memory.SharedMemory(create=True, size=detection_array.nbytes,name='detection')
    # detection = np.ndarray(detection_array.shape, dtype=detection_array.dtype, buffer=shm_detect.buf)
    # detection[:] = detection_array[:]

    detection_array_all = np.zeros((6,4), dtype=np.int32)
    shm_detect_all = shared_memory.SharedMemory(create=True, size=detection_array_all.nbytes,name='detection_all')
    detection_all = np.ndarray(detection_array_all.shape, dtype=detection_array_all.dtype, buffer=shm_detect_all.buf)
    detection_all[:] = detection_array_all[:]    
  
    
    voice_rec_arr_dummie = np.empty(1, dtype='<U512')
    shm_voice_rec = shared_memory.SharedMemory(create=True, size=voice_rec_arr_dummie.nbytes,name='voice_detect')
    voice_rec_array = np.ndarray(voice_rec_arr_dummie.shape, dtype=voice_rec_arr_dummie.dtype, buffer=shm_voice_rec.buf)
    voice_rec_array[:] = voice_rec_arr_dummie[:]

    for frame in tello.container.decode(video=0):
        
        if 0 < frame_skip:
            frame_skip = frame_skip - 1
            continue
        start_time = time.time()
        if frame.time_base < 1.0/60:
            time_base = 1.0/60
        else:
            time_base = frame.time_base
        
        image = cv2.cvtColor(np.array(frame.to_image(),dtype=np.uint8), cv2.COLOR_RGB2BGR)
        
        if first_frame:
                
                    # Create the shared memory to share the current frame decoded by the parent process 
                    # and given to the child process for further processing 
                    shm = shared_memory.SharedMemory(create=True, size=image.nbytes,name='img_to_yolo')
                    shared_img_to_yolo = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
                    shared_img_to_yolo[:] = image[:] 

                    shm_from_yolo = shared_memory.SharedMemory(create=True, size=image.nbytes,name='img_from_yolo')
                    shared_img_from_yolo = np.ndarray(image.shape, dtype=image.dtype, buffer=shm_from_yolo.buf)
                                  
                    first_frame = False
                    
                    # Launch process child
                    p_worker = Process(target=detect_objects)
                    p_worker.daemon = True
                    p_worker.start()

                    #Launch Speech Listener
                    p_voice = Process(target=start_rhino, daemon=True)
                    p_voice.start()
                    time.sleep(3)
               
        #Write the current frame in shared memory to Yolo       
        shared_img_to_yolo[:] = image[:]

        #Get the current frame in shared memory from Yolo      
        image[:] = shared_img_from_yolo[:] 

        #Get Yolo Head Detections from Child process
        # target = (detection[0], detection[1])
        # target_width = detection[2]
        # target_height = detection[3]
        
        #Get Voice Commands
        tello.process_audio(voice_rec_array)
        
        #Process Frame and change Tello Commands
        image = tello.process_frame(image, detection_all)

        tello.fps.update()

        # Display the frame
        cv2.imshow('Tello', image)
        key = cv2.waitKey(1) & 0xff

        frame_skip = int((time.time() - start_time)/time_base)
    

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-l","--log_level", help="select a log level (info, debug)")
    #ap.add_argument("-2","--multiprocess", action='store_true', help="use 2 processes to share the workload (instead of 1)")
    args=ap.parse_args()

    main(log_level=args.log_level)