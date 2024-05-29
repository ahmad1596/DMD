from pyvcam import pvc
from pyvcam.camera import Camera
import matplotlib.pyplot as plt
import time
import os
from PIL import Image

def capture_frames(duration, interval, save_dir):
    pvc.init_pvcam()
    camera = next(Camera.detect_camera())
    camera.open()
    start_time = time.time()
    end_time = start_time + duration
    frames = []
    while time.time() < end_time:
        frame = camera.get_frame(exp_time=exposure_time)
        frames.append(frame)
        time.sleep(interval)
    camera.close()
    pvc.uninit_pvcam()
    print(f"Total images acquired: {len(frames)}")
    print(f"Duration: {duration} seconds")
    print(f"Interval: {interval} seconds")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, frame in enumerate(frames):
        image = Image.fromarray(frame)
        image_path = os.path.join(save_dir, f"frame_{i+1}.tiff")
        image.save(image_path)
        print(f"Saved frame {i+1} as {image_path}")
        plt.figure(dpi=600)
        plt.imshow(frame, cmap='gray')
        plt.title(f"Frame {i+1}")
        plt.show()
exposure_time = 10  # Exposure time in milliseconds
duration = 3  # Duration of capture in seconds
interval = 1  # Time interval between captures in seconds
save_dir = r"C:\Users\DELL\Documents\saved_frames_python"

capture_frames(duration, interval, save_dir)
