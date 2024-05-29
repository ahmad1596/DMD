from pyvcam import pvc
from pyvcam.camera import Camera
import numpy as np
import cv2

def main():
    pvc.init_pvcam()
    try:
        cam = next(Camera.detect_camera())
        cam.open()
        exposure_time = 10
        cam.start_live(exp_time=exposure_time)
        while True:
            frame_data = cam.poll_frame(copyData=True)
            frame = frame_data[0]['pixel_data']
            frame = frame.astype(np.uint16)
            normalized_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame_uint8 = normalized_frame.astype(np.uint8)
            resized_frame = cv2.resize(frame_uint8, (300, 300))
            frame_bgr = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Live View', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.finish()
        cam.close()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
