import time
import datetime
import os
import tellopy
import numpy as np
import av
import cv2
from pynput import keyboard
from threading import Thread
import argparse
from math import pi, atan2, degrees, sqrt
from simple_pid import PID
from FPS import FPS
from CameraMorse import RollingGraph
import logging
import re


def quat_to_yaw_deg(qx,qy,qz,qw):
    """
        Calculate yaw from quaternion
    """
    degree = pi/180
    sqy = qy*qy
    sqz = qz*qz
    siny = 2 * (qw*qz+qx*qy)
    cosy = 1 - 2*(qy*qy+qz*qz)
    yaw = int(atan2(siny,cosy)/degree)
    return yaw


log = logging.getLogger("TelloYolo")

class TelloController(object):
    """
    TelloController builds keyboard controls on top of TelloPy as well
    as generating images from the video stream and enabling opencv support
    """

    def __init__(self, use_face_tracking=True, 
                kbd_layout="QWERTY", 
                write_log_data=False, 
                media_directory="media", 
                child_cnx=None,
                log_level='info'):
        
        self.pid_pitch = PID(0.3,0.0,0.15,setpoint=0,output_limits=(-80,80))
        self.pid_throttle = PID(0.3,0.0,0.02,setpoint=0,output_limits=(-80,100))
        self.pid_yaw = PID(0.18,0.0,0.01,setpoint=0,output_limits=(-100,100))
        self.pid_roll = PID(0.2,0.0,0.15,setpoint=0,output_limits=(-30,30))
        self.log_level = log_level
        self.debug = log_level is not None
        

        # Flight data
        self.is_flying = False
        self.battery = None
        self.fly_mode = None
        self.throw_fly_timer = 0
        self.manual_control = False
        self.tracking_after_takeoff = False
        self.record = False
        self.keydown = False
        self.date_fmt = '%Y-%m-%d_%H%M%S'
        self.kbd_layout = kbd_layout

        # Init Drone 
        self.drone = tellopy.Tello()
        self.axis_command = {
            "yaw": self.drone.clockwise,
            "roll": self.drone.right,
            "pitch": self.drone.forward,
            "throttle": self.drone.up
        }
        self.axis_speed = { "yaw":0, "roll":0, "pitch":0, "throttle":0}
        self.cmd_axis_speed = { "yaw":0, "roll":0, "pitch":0, "throttle":0}
        self.prev_axis_speed = self.axis_speed.copy()
        self.def_speed =  { "yaw":50, "roll":35, "pitch":35, "throttle":80}     
        self.move_timestamp = 2
        self.search_start_time = None
        self.write_log_data = write_log_data
        self.reset()
        self.media_directory = media_directory
        if not os.path.isdir(self.media_directory):
            os.makedirs(self.media_directory)

        if self.write_log_data:
            path = 'tello-%s.csv' % datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
            self.log_file = open(path, 'w')
            self.write_header = True

        self.init_drone()
        self.init_controls()
        

        #Voice Command
        self.prev_voice_command = ''
        self.use_voice = False
        self.picture_target = None
        self.picture_approach = False
        self.rth = False
        self.timestamp_pic_target_found = None
        self.classes_dict = {
                        'orange' : 0,
                        'person' : 1,
                        'guitar' : 2,
                        'human hand' : 3,
                        'me' : 4,
                        'chair' : 5,
        }

        # container for processing the packets into frames
        # self.container = av.open(self.drone.get_video_stream())
        # self.vid_stream = self.container.streams.video[0]
        self.container = None
        self.collect_frames = True
        self.target_height = None
        self.prev_target_heigth = None
        self.frame_shape = None
        self.out_file = None
        self.out_stream = None
        self.out_name = None
        self.start_time = time.time()
        self.is_pressed = False
        self.fps = FPS()
        self.exposure = 0

                            
        # Logging
        if log_level is not None:
            if log_level == "info":
                log_level = logging.INFO
            elif log_level == "debug":
                log_level = logging.DEBUG
            log.setLevel(log_level)
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(log_level)
            ch.setFormatter(logging.Formatter(fmt='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S"))
            log.addHandler(ch)
    
    def set_video_encoder_rate(self, rate):
        self.drone.set_video_encoder_rate(rate)
        self.video_encoder_rate = rate

    def streamer(self):
        """ 
        Fill video container with frames from video stream
        """
        retry = 3
        print ('start streamer!')
        while self.container is None and 0 < retry:
            if not self.collect_frames:
                break
            #print (type(self.container))
            retry -= 1
            try:
                self.container = av.open(self.drone.get_video_stream())
                print('success')
            except av.AVError as ave:
                print(ave)
                print('retry...')
    
    def get_frames(self):
        """ start child process for video encoding"""
        video_getter = Thread(target=self.streamer)
        video_getter.daemon = True
        video_getter.start()

    def reset (self):
        """
            Reset global variables before a fly
        """
        log.debug("RESET")
        self.ref_pos_x = -1
        self.ref_pos_y = -1
        self.ref_pos_z = -1
        self.pos_x = -1
        self.pos_y = -1
        self.pos_z = -1
        self.yaw = 0
        self.tracking = False
        self.distance_mode = False
        self.keep_distance = None
        self.hand_ctrl = False
        self.timestamp_hand_ctrl = None
        self.head_hand_x_ref = None
        self.head_hand_x_dist = None
        self.palm_landing = False
        self.palm_landing_approach = False
        self.yaw_to_consume = 0
        self.timestamp_keep_distance = time.time()
        self.wait_before_tracking = None
        self.timestamp_take_picture = None
        self.throw_ongoing = False
        self.scheduled_takeoff = None

        # When in trackin mode, but no body is detected in current frame,
        # we make the drone rotate in the hope to find some body
        # The rotation is done in the same direction as the last rotation done
        self.body_in_prev_frame = False
        self.timestamp_no_body = time.time()
        self.last_rotation_is_cw = True

    def init_drone(self):
        """
            Connect to the drone, start streaming and subscribe to events
        """
        #if self.log_level:
        self.drone.log.set_level(0)
        self.drone.connect()
        self.set_video_encoder_rate(3)
        self.drone.start_video()

        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA,
                             self.flight_data_handler)
        self.drone.subscribe(self.drone.EVENT_LOG_DATA,
                             self.log_data_handler)
        self.drone.subscribe(self.drone.EVENT_FILE_RECEIVED,
                             self.handle_flight_received)

    def on_press(self, keyname):
        """
            Handler for keyboard listener
        """
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip('\'')
            log.info('KEY PRESS ' + keyname)
            if keyname == 'Key.esc':
                self.toggle_tracking(False)
                # self.tracking = False
                self.drone.land()
                self.drone.quit()

                
                cv2.destroyAllWindows() 
                os._exit(0)
            
            if keyname in self.controls_keypress:
                self.controls_keypress[keyname]()
        except AttributeError:
            log.debug(f'special key {keyname} pressed')
    
    def on_release(self, keyname):
        """
            Reset on key up from keyboard listener
        """
        self.keydown = False
        keyname = str(keyname).strip('\'')
        log.info('KEY RELEASE ' + keyname)
        if keyname in self.controls_keyrelease:
            key_handler = self.controls_keyrelease[keyname]()
    
    def set_speed(self, axis, speed):
        """Set Tello Axis Speed"""
        #log.info(f"set speed {axis} {speed}")
        self.cmd_axis_speed[axis] = speed
        
    def init_controls(self):
        """
            Define keys and add listener
        """


        controls_keypress_QWERTY = {
            'w': lambda: self.set_speed("pitch", self.def_speed["pitch"]),
            's': lambda: self.set_speed("pitch", -self.def_speed["pitch"]),
            'a': lambda: self.set_speed("roll", -self.def_speed["roll"]),
            'd': lambda: self.set_speed("roll", self.def_speed["roll"]),
            'q': lambda: self.set_speed("yaw", -self.def_speed["yaw"]),
            'e': lambda: self.set_speed("yaw", self.def_speed["yaw"]),
            'i': lambda: self.drone.flip_forward(),
            'k': lambda: self.drone.flip_back(),
            'j': lambda: self.drone.flip_left(),
            'l': lambda: self.drone.flip_right(),
            'Key.left': lambda: self.set_speed("yaw", -1.5*self.def_speed["yaw"]),
            'Key.right': lambda: self.set_speed("yaw", 1.5*self.def_speed["yaw"]),
            'Key.up': lambda: self.set_speed("throttle", self.def_speed["throttle"]),
            'Key.down': lambda: self.set_speed("throttle", -self.def_speed["throttle"]),
            'Key.tab': lambda: self.drone.takeoff(),
            'Key.backspace': lambda: self.drone.land(),
            'p': lambda: self.palm_land_approach(),
            'v': lambda: self.toggle_use_voice(),
            't': lambda: self.toggle_tracking(),
            'k': lambda: self.toggle_distance_mode(),
            'm': lambda: self.toogle_manual_control(),
            'Key.enter': lambda: self.take_picture(),
            'c': lambda: self.clockwise_degrees(360),
            
            
            
            
            
            
            # '0': lambda: self.drone.set_video_encoder_rate(0),
            # '1': lambda: self.drone.set_video_encoder_rate(1),
            # '2': lambda: self.drone.set_video_encoder_rate(2),
            # '3': lambda: self.drone.set_video_encoder_rate(3),
            # '4': lambda: self.drone.set_video_encoder_rate(4),
            # '5': lambda: self.drone.set_video_encoder_rate(5),

            '7': lambda: self.set_exposure(-1),    
            '8': lambda: self.set_exposure(0),
            '9': lambda: self.set_exposure(1)
        }

        controls_keyrelease_QWERTY = {
            'w': lambda: self.set_speed("pitch", 0),
            's': lambda: self.set_speed("pitch", 0),
            'a': lambda: self.set_speed("roll", 0),
            'd': lambda: self.set_speed("roll", 0),
            'q': lambda: self.set_speed("yaw", 0),
            'e': lambda: self.set_speed("yaw", 0),
            'Key.left': lambda: self.set_speed("yaw", 0),
            'Key.right': lambda: self.set_speed("yaw", 0),
            'Key.up': lambda: self.set_speed("throttle", 0),
            'Key.down': lambda: self.set_speed("throttle", 0)
        }

        controls_keypress_AZERTY = {
            'z': lambda: self.set_speed("pitch", self.def_speed["pitch"]),
            's': lambda: self.set_speed("pitch", -self.def_speed["pitch"]),
            'q': lambda: self.set_speed("roll", -self.def_speed["roll"]),
            'd': lambda: self.set_speed("roll", self.def_speed["roll"]),
            'a': lambda: self.set_speed("yaw", -self.def_speed["yaw"]),
            'e': lambda: self.set_speed("yaw", self.def_speed["yaw"]),
            'i': lambda: self.drone.flip_forward(),
            'k': lambda: self.drone.flip_back(),
            'j': lambda: self.drone.flip_left(),
            'l': lambda: self.drone.flip_right(),
            'Key.left': lambda: self.set_speed("yaw", -1.5*self.def_speed["yaw"]),
            'Key.right': lambda: self.set_speed("yaw", 1.5*self.def_speed["yaw"]),
            'Key.up': lambda: self.set_speed("throttle", self.def_speed["throttle"]),
            'Key.down': lambda: self.set_speed("throttle", -self.def_speed["throttle"]),
            'Key.tab': lambda: self.drone.takeoff(),
            'Key.backspace': lambda: self.drone.land(),
            'p': lambda: self.palm_land(),
            't': lambda: self.toggle_tracking(),
            'Key.enter': lambda: self.take_picture(),
            'c': lambda: self.clockwise_degrees(360),
            '0': lambda: self.drone.set_video_encoder_rate(0),
            '1': lambda: self.drone.set_video_encoder_rate(1),
            '2': lambda: self.drone.set_video_encoder_rate(2),
            '3': lambda: self.drone.set_video_encoder_rate(3),
            '4': lambda: self.drone.set_video_encoder_rate(4),
            '5': lambda: self.drone.set_video_encoder_rate(5),

            '7': lambda: self.set_exposure(-1),    
            '8': lambda: self.set_exposure(0),
            '9': lambda: self.set_exposure(1)
        }

        controls_keyrelease_AZERTY = {
            'z': lambda: self.set_speed("pitch", 0),
            's': lambda: self.set_speed("pitch", 0),
            'q': lambda: self.set_speed("roll", 0),
            'd': lambda: self.set_speed("roll", 0),
            'a': lambda: self.set_speed("yaw", 0),
            'e': lambda: self.set_speed("yaw", 0),
            'Key.left': lambda: self.set_speed("yaw", 0),
            'Key.right': lambda: self.set_speed("yaw", 0),
            'Key.up': lambda: self.set_speed("throttle", 0),
            'Key.down': lambda: self.set_speed("throttle", 0)
        }

        if self.kbd_layout == "AZERTY":
            self.controls_keypress = controls_keypress_AZERTY
            self.controls_keyrelease = controls_keyrelease_AZERTY
        else:
            self.controls_keypress = controls_keypress_QWERTY
            self.controls_keyrelease = controls_keyrelease_QWERTY
        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()

    def process_audio(self, voice_rec_array):
        """ Get and process Voice Commands
        """
        
        if self.use_voice:
            
            voice = voice_rec_array[0]
        
            if self.prev_voice_command != voice:
                
                print(f'voice command: {voice}')

                if 'takeoff' in voice:
                    self.drone.takeoff() 
                    print('takeoff')

                if 'land' in voice:
                    if 'palm' in voice:
                        print('palmland')
                        self.palm_land_approach()

                    else:
                        self.toggle_tracking(False)
                        # self.tracking = False
                        self.drone.land()
                        self.drone.quit()
                        cv2.destroyAllWindows() 
                        os._exit(0)
                    
                
                if 'tracking' in voice:
                    if 'no' in voice:
                        self.tracking = False
                    else:
                        self.tracking = True


                if 'distance' in voice:
                    if 'off' in voice:
                        self.distance_mode = False
                        self.keep_distance = None
                    else:
                        self.distance_mode = True
                    

                if 'picture' in voice:
                    print('picture in command')
                    self.picture_target = re.findall("pic_target\s:\s'([a-zA-Z\s]+)",voice)[0].replace('the ','')
                    self.toggle_tracking(tracking=False)
                    print('tracking off')
                    self.picture_approach = True
                    self.target_height = None
                    self.search_start_time = time.time()
                
                if 'come' in voice:
                    self.rth = True

                self.prev_voice_command = voice

                # if 'move' in voice:
                #     amount = None
                #     amount = int(re.findall('[0-9]{2}', voice)[0])
                #     print(amount)

                    # if amount is not None:

                    #     if amount > 30:
                    #         amount = 30
                    #     try:
                    #         if 'forwards' in voice:
                    #             self.drone.forward(amount)
                            
                    #         if 'backwards' in voice:
                    #             self.drone.backward(amount)

                    #         if 'left' in voice:
                    #             self.drone.left(amount)

                    #         if 'right' in voice:
                    #             self.drone.right(amount)

                    #         self.move_timestamp = time.time()


                    #     except:
                    #         print('not possible')
                        
                    # self.prev_voice_command = voice
             
    def process_frame(self, raw_frame, detection_all):
        """
            Analyze the frame and return the frame with information (HUD, target bbox) drawn on it
        """
        
        frame = raw_frame.copy()
        h,w,_ = frame.shape
        proximity = int(h/2.4)
        pic_proximity = int(h/1.45)
        min_distance = int(w/2)

        head_pos = (detection_all[4][0], detection_all[4][1])
        hand_pos = (detection_all[3][0], detection_all[3][1])
        
        
        #self.target_height = target_height
        self.target_height = detection_all[4][3]
        target_width = detection_all[4][2]
        target = (detection_all[4][0], detection_all[4][1])

        ref_x = int(w/2)
        ref_y = int(h*0.35)
        
        self.axis_speed = self.cmd_axis_speed.copy()

        #Is there a Picture Command ?
        if self.picture_approach:
            cls_number = int(self.classes_dict[self.picture_target])
            print (str(self.picture_target) + 'is' + str(cls_number))
            print (self.picture_target + ' values:' + str(detection_all[cls_number]))
            
            # If no pic target found --> rotate
            if (detection_all[cls_number][0] + detection_all[cls_number][1]) == 0:
                
                log.info(f'searching for {self.picture_target}')
                
                if time.time() - self.search_start_time < 8: 
                    self.axis_speed["yaw"] = 60
                else:
                    print('stopped searching after 8 seconds')
                    self.axis_speed["yaw"] = 0
                    self.picture_approach = False
            
            # If pic target found set as new tracking target
            else:
                print('pic target found')
                self.axis_speed["yaw"] = 0
                if self.timestamp_pic_target_found is None:
                    self.timestamp_pic_target_found = time.time()

                log.info(f'found {self.picture_target}')
                target = (detection_all[cls_number][0], detection_all[cls_number][1])
                self.target_height = detection_all[cls_number][3]
                
                #If Human Head:
                if cls_number == 4:
                    self.keep_distance = pic_proximity*0.75
                else:
                    self.keep_distance = pic_proximity

                self.pid_pitch = PID(0.15,0.0,0.1,setpoint=0,output_limits=(-30,30))
                self.tracking = True
        
        # If voice command 'come home' activate RTH
        if self.rth:
            self.target_height = detection_all[4][3]
            target_width = detection_all[4][2]
            target = (detection_all[4][0], detection_all[4][1])
            self.keep_distance = proximity*0.75
            self.toggle_tracking(tracking=True)

        if self.timestamp_take_picture:
            if time.time() - self.timestamp_take_picture > 2:
                self.timestamp_take_picture = None
                self.drone.take_picture()
        else:

            if self.tracking:                        
                if target != (0,0):  
                    if self.distance_mode:  
                    # Locked distance mode
                        if self.keep_distance is None:
                            self.keep_distance = self.target_height
                            #self.graph_distance = RollingGraph(window_name="Distance", y_max=500, threshold=self.keep_distance, waitKey=False)
                    
                    if self.palm_landing_approach:
                        self.keep_distance = proximity
                        self.timestamp_keep_distance = time.time()
                        log.info("APPROACHING on pose")
                        self.pid_pitch = PID(0.2,0.0,0.1,setpoint=0,output_limits=(-30,30))
                        #self.graph_distance = RollingGraph(window_name="Distance", y_max=500, threshold=self.keep_distance, waitKey=False)

                    self.body_in_prev_frame = True
                       
                    xoff = int(target[0]-ref_x)
                    yoff = int(ref_y-target[1])

                    #We draw an arrow from the reference point to the head we are targeting    
                    color = (0,0,255)
                    cv2.circle(frame, (ref_x, ref_y), 10, color, 1,cv2.LINE_AA)
                    cv2.line(frame, (ref_x, ref_y), target, color, 4)
                    cv2.rectangle(frame, (target[0]-target_width//2, target[1]-self.target_height//2), 
                                         (target[0]+target_width//2, target[1]+self.target_height//2),color,4)
                    
                    # The PID controllers calculate the new speeds for yaw and throttle
                    self.axis_speed["yaw"] = int(-self.pid_yaw(xoff))
                    #log.debug(f"xoff: {xoff} - speed_yaw: {self.axis_speed['yaw']}")
                    self.last_rotation_is_cw = self.axis_speed["yaw"] > 0

                    self.axis_speed["throttle"] = int(-self.pid_throttle(yoff))
                    #log.debug(f"yoff: {yoff} - speed_throttle: {self.axis_speed['throttle']}")

                    #If in locked distance mode
                    if self.keep_distance and self.target_height:   
                        
                        # Check RTH
                        if self.rth and self.target_height>=self.keep_distance:
                            self.rth = False
                        
                        elif self.palm_landing_approach and self.target_height>self.keep_distance:
                            # The drone is now close enough to the body
                            # Let's do the palm landing
                            log.info("PALM LANDING after approaching")
                            self.palm_landing_approach = False
                            self.toggle_tracking(tracking=False)
                            self.palm_land() 
                        
                        elif self.picture_approach and \
                            abs(self.target_height-self.keep_distance) < 15 and \
                            xoff < 12 and yoff < 15:
                            
                            # The drone is now close enough to the pic target
                            # Let's take a picture 
                            self.toggle_tracking(tracking=False)
                            print('take a picture')
                            self.drone.take_picture()
                            self.picture_approach = False
                            self.timestamp_pic_target_found = None
                            self.pid_pitch = PID(0.3,0.0,0.1,setpoint=0,output_limits=(-70,70))
                        
 
                        else:
                            self.axis_speed["pitch"] = int(self.pid_pitch(self.target_height-self.keep_distance))
                            log.debug(f"Target distance: {self.keep_distance} - cur: {self.target_height} -speed_pitch: {self.axis_speed['pitch']}")
                            
                            if abs(head_pos[1] - hand_pos[1])<30:
                                if self.timestamp_hand_ctrl is None:
                                    self.timestamp_hand_ctrl = time.time()
                                if time.time() - self.timestamp_hand_ctrl > 1:
                                    if self.head_hand_x_dist is None:
                                        self.head_hand_x_ref = head_pos[0]-hand_pos[0]
                                
                                    self.hand_ctrl = True
                                    self.head_hand_x_dist = head_pos[0]-hand_pos[0]
                                    self.axis_speed["roll"] = int(-self.pid_roll(self.head_hand_x_ref - self.head_hand_x_dist))
                                    #print (f'head hand X distance: {abs(head_pos[0]-hand_pos[0])}')

                            else:
                                self.hand_ctrl = False
                                self.timestamp_hand_ctrl = None
                                self.head_hand_x_dist = None

                else: # Tracking but no body detected
                    if self.body_in_prev_frame:
                        self.timestamp_no_body = time.time()
                        self.body_in_prev_frame = False
                        self.axis_speed["throttle"] = self.prev_axis_speed["throttle"]
                        self.axis_speed["yaw"] = self.prev_axis_speed["yaw"]
                    else:
                        if time.time() - self.timestamp_no_body < 1:
                            print("NO BODY SINCE < 1", self.axis_speed, self.prev_axis_speed)
                            self.axis_speed["throttle"] = self.prev_axis_speed["throttle"]
                            self.axis_speed["yaw"] = self.prev_axis_speed["yaw"]
                        else:
                            log.debug("NO BODY detected for 1s -> rotate")
                            self.axis_speed["yaw"] = self.def_speed["yaw"] * (1 if self.last_rotation_is_cw else -1)
                

        # Send axis commands to the drone
        for axis, command in self.axis_command.items():
            if self.axis_speed[axis]is not None and self.axis_speed[axis] != self.prev_axis_speed[axis]:
                #log.debug(f"COMMAND {axis} : {self.axis_speed[axis]}")
                command(self.axis_speed[axis])
                self.prev_axis_speed[axis] = self.axis_speed[axis]
            else:
                # This line is necessary to display current values in 'self.write_hud'
                self.axis_speed[axis] = self.prev_axis_speed[axis]
        
        # Write the HUD on the frame
        frame = self.write_hud(frame)

        return frame

    def write_hud(self, frame):
        """
            Draw drone info on frame
        """

        class HUD:
            def __init__(self, def_color=(255, 170, 0)):
                self.def_color = def_color
                self.infos = []
            def add(self, info, color=None):
                if color is None: color = self.def_color
                self.infos.append((info, color))
            def draw(self, frame):
                i=0
                for (info, color) in self.infos:
                    cv2.putText(frame, info, (0, 30 + (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2) #lineType=30)
                    i+=1
                

        hud = HUD()
        tello_color = (0,255,0)
        if self.debug: hud.add(datetime.datetime.now().strftime('%H:%M:%S'))
        hud.add(f"FPS {self.fps.get():.2f}")
        if self.debug: hud.add(f"VR {self.video_encoder_rate}")

        hud.add(f"BAT {self.battery}")
        if self.is_flying:
            hud.add("FLYING", (0,255,0))
        else:
            hud.add("NOT FLYING", (0,0,255))
        hud.add(f"TRACKING {'ON' if self.tracking else 'OFF'}", (0,255,0) if self.tracking else (0,0,255) )
        hud.add(f"EXPO {self.exposure}")
        #hud.add(f"ALT {self.ref_pos_x}")
        

        if self.hand_ctrl:
            hud.add(f"HAND Ctrl {self.ref_pos_x}",(0,0,255))
            hud.add(f"HEAD_HAND_DIST {self.head_hand_x_ref - self.head_hand_x_dist}")
        if self.axis_speed['yaw'] > 0:
            hud.add(f"CW {self.axis_speed['yaw']}", tello_color)
        elif self.axis_speed['yaw'] < 0:
            hud.add(f"CCW {-self.axis_speed['yaw']}", tello_color)
        else:
            hud.add(f"CW 0")
        if self.axis_speed['roll'] > 0:
            hud.add(f"RIGHT {self.axis_speed['roll']}", tello_color)
        elif self.axis_speed['roll'] < 0:
            hud.add(f"LEFT {-self.axis_speed['roll']}", tello_color)
        else:
            hud.add(f"RIGHT 0")
        if self.axis_speed['pitch'] > 0:
            hud.add(f"FORWARD {self.axis_speed['pitch']}", tello_color)
        elif self.axis_speed['pitch'] < 0:
            hud.add(f"BACKWARD {-self.axis_speed['pitch']}", tello_color)
        else:
            hud.add(f"FORWARD 0")
        if self.axis_speed['throttle'] > 0:
            hud.add(f"UP {self.axis_speed['throttle']}", tello_color)
        elif self.axis_speed['throttle'] < 0:
            hud.add(f"DOWN {-self.axis_speed['throttle']}",tello_color)
        else:
            hud.add(f"UP 0")
        if self.keep_distance: 
            hud.add(f"Target distance: {self.keep_distance} - curr: {self.target_height}", (0,255,0))
            #if self.target_height: self.graph_distance.new_iter([self.target_height])
        if self.timestamp_take_picture: hud.add("Taking a picture",tello_color)
        if self.palm_landing:
            hud.add("Palm landing...", tello_color)
        if self.palm_landing_approach:
            hud.add("In approach for palm landing...", tello_color)
        if self.tracking and not self.body_in_prev_frame and time.time() - self.timestamp_no_body > 0.5:
            hud.add("Searching...", tello_color)
        if self.manual_control:
            hud.add("Manual Control...", tello_color)
        if self.throw_ongoing:
            hud.add("Throw ongoing...", tello_color)
        if self.scheduled_takeoff:
            seconds_left = int(self.scheduled_takeoff - time.time())
            hud.add(f"Takeoff in {seconds_left}s")

        hud.draw(frame)
        
        return frame

    def take_picture(self):
        """
            Tell drone to take picture, image sent to file handler
        """
        self.drone.take_picture()

    def set_exposure(self, expo):
        """
            Change exposure of drone camera
        """
        if expo == 0:
            self.exposure = 0
        elif expo == 1:
            self.exposure = min(9, self.exposure+1)
        elif expo == -1:
            self.exposure = max(-9, self.exposure-1)
        self.drone.set_exposure(self.exposure)
        log.info(f"EXPOSURE {self.exposure}")

    def palm_land_approach(self):
            self.palm_landing_approach = True
            print('Palm Landing Approach')

    def palm_land(self):
        """
            Tell drone to land
        """
        self.palm_landing = True
        self.drone.palm_land()

    def throw_and_go(self, tracking=False):
        """
            Tell drone to start a 'throw and go'
        """
        self.drone.throw_and_go()      
        self.tracking_after_takeoff = tracking
        
    def delayed_takeoff(self, delay=5):
        self.scheduled_takeoff = time.time()+delay
        self.tracking_after_takeoff = True
        
    def clockwise_degrees(self, degrees):
        self.yaw_to_consume = degrees
        self.yaw_consumed = 0
        self.prev_yaw = self.yaw
        
    def toggle_distance_mode(self):
        self.distance_mode = not self.distance_mode

        if not self.distance_mode:
            self.keep_distance = None
        log.info('distance_mode '+("ON" if self.distance_mode else "OFF"))

    def toggle_use_voice(self):
        self.use_voice = not self.use_voice
        log.info('use_voice '+("ON" if self.use_voice else "OFF"))

    def toogle_manual_control(self):
        self.manual_control = not self.manual_control
        if self.manual_control:    
            self.axis_speed = { "yaw":0, "roll":0, "pitch":0, "throttle":0}
            self.keep_distance = None
        log.info('manual_control '+("ON" if self.manual_control else "OFF"))
         
    def toggle_tracking(self, tracking=None):
        """ 
            If tracking is None, toggle value of self.tracking
            Else self.tracking take the same value as tracking
        """
        
        if tracking is None:
            self.tracking = not self.tracking
        else:
            self.tracking = tracking
        if self.tracking:
            log.info("ACTIVATE TRACKING")


            # Start an explarotary 360
            #self.clockwise_degrees(360)
            # Init a PID controller for the yaw
            #self.pid_yaw = PID(0.25,0,0,setpoint=0,output_limits=(-100,100))
            # ... and one for the throttle
            #self.pid_throttle = PID(0.4,0,0,setpoint=0,output_limits=(-80,100))
            # self.init_tracking = True
        else:
            self.axis_speed = { "yaw":0, "roll":0, "pitch":0, "throttle":0}
            self.keep_distance = None

        return

    def flight_data_handler(self, event, sender, data):
        """
            Listener to flight data from the drone.
        """
        self.battery = data.battery_percentage
        self.fly_mode = data.fly_mode
        self.throw_fly_timer = data.throw_fly_timer
        self.throw_ongoing = data.throw_fly_timer > 0

        # print("fly_mode",data.fly_mode)
        # print("throw_fly_timer",data.throw_fly_timer)
        # print("em_ground",data.em_ground)
        # print("em_sky",data.em_sky)
        # print("electrical_machinery_state",data.electrical_machinery_state)
        #print("em_sky",data.em_sky,"em_ground",data.em_ground,"em_open",data.em_open)
        #print("height",data.height,"imu_state",data.imu_state,"down_visual_state",data.down_visual_state)
        if self.is_flying != data.em_sky:            
            self.is_flying = data.em_sky
            log.debug(f"FLYING : {self.is_flying}")
            if not self.is_flying:
                self.reset()
            else:
                if self.tracking_after_takeoff:
                    log.info("Tracking on after takeoff")
                    self.toggle_tracking(True)
                    
        log.debug(f"MODE: {self.fly_mode} - Throw fly timer: {self.throw_fly_timer}")

    def log_data_handler(self, event, sender, data):
        """
            Listener to log data from the drone.
        """  

        pos_x = -data.mvo.pos_x
        pos_y = -data.mvo.pos_y
        pos_z = -data.mvo.pos_z

        self.ref_pos_x = pos_x
        #print(f'pos x = {pos_x}, pos y = {pos_y}, pos z = {pos_z}')

        # if abs(pos_x)+abs(pos_y)+abs(pos_z) > 0.07:
        #     if self.ref_pos_x == -1: # First time we have meaningful values, we store them as reference
        #         self.ref_pos_x = pos_x
        #         self.ref_pos_y = pos_y
        #         self.ref_pos_z = pos_z
        #     else:
        #         self.pos_x = pos_x - self.ref_pos_x
        #         self.pos_y = pos_y - self.ref_pos_y
        #         self.pos_z = pos_z - self.ref_pos_z
        
        qx = data.imu.q1
        qy = data.imu.q2
        qz = data.imu.q3
        qw = data.imu.q0
        self.yaw = quat_to_yaw_deg(qx,qy,qz,qw)
        #print(f'yaw = {self.yaw}')
        
        if self.write_log_data:
            if self.write_header:
                self.log_file.write('%s\n' % data.format_cvs_header())
                self.write_header = False
            self.log_file.write('%s\n' % data.format_cvs())

    def handle_flight_received(self, event, sender, data):
        """
            Create a file in local directory to receive image from the drone
        """
        path = f'{self.media_directory}/tello-{datetime.datetime.now().strftime(self.date_fmt)}.jpg' 
        with open(path, 'wb') as out_file:
            out_file.write(data)
        log.info('Saved photo to %s' % path)
