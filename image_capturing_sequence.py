# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
from pyueye import ueye
import numpy as np
import cv2
import os

def create_output_directory(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Output directory '{output_directory}' created.")

def set_and_get_pixel_clock(hCam, desired_pixel_clock=None):
    if desired_pixel_clock is not None:
        nRet = ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_SET, ueye.c_uint(desired_pixel_clock), ueye.sizeof(ueye.c_uint(desired_pixel_clock)))
        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock (Set) ERROR")
            return None
    pixel_clock = ueye.c_uint()
    nRet = ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_GET, pixel_clock, ueye.sizeof(pixel_clock))
    if nRet != ueye.IS_SUCCESS:
        print("is_PixelClock (Get) ERROR")
        return None
    return pixel_clock.value

def set_and_get_exposure_time(hCam, exposure_time=None):
    if exposure_time is not None:
        nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.c_double(exposure_time), ueye.sizeof(ueye.c_double(exposure_time)))
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure (Set) ERROR")
            return None
    exposure_time = ueye.c_double()
    nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exposure_time, ueye.sizeof(exposure_time))
    if nRet != ueye.IS_SUCCESS:
        print("is_Exposure (Get) ERROR")
        return None
    return exposure_time.value

def setup_camera():
    hCam = ueye.HIDS(0)
    nRet = ueye.is_InitCamera(hCam, None)
    if nRet != ueye.IS_SUCCESS:
        print("is_InitCamera ERROR")
        return None
    return hCam

def activate_camera_and_setup_image_memory(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch):
    nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")
        return False
    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
    if nRet != ueye.IS_SUCCESS:
        print("is_InquireImageMem ERROR")
        return False
    return True

def set_master_gain_to_zero(hCam):
    enable_auto_gain = 0.0
    ret = ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.c_double(enable_auto_gain), None)
    if ret != ueye.IS_SUCCESS:
        print("Error setting auto gain control")

def print_camera_settings_info(sInfo, rectAOI, width, height, nBitsPerPixel):
    print("\nCamera Information:")
    print(f"  Maximum Width: {sInfo.nMaxWidth} pixels")
    print(f"  Maximum Height: {sInfo.nMaxHeight} pixels")
    print("\nAOI (Area of Interest):")
    print(f"  X Position: {rectAOI.s32X}")
    print(f"  Y Position: {rectAOI.s32Y}")
    print(f"  Width: {width} pixels")
    print(f"  Height: {height} pixels")
    print("\nSaved Captured Image Information:")
    print(f"  Width: {width} pixels")
    print(f"  Height: {height} pixels")
    print(f"  Bits Per Pixel: {nBitsPerPixel} bits")

def cleanup_camera(hCam, pcImageMemory, MemID):
    ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)
    ueye.is_ExitCamera(hCam)
    cv2.destroyAllWindows()

def capture_and_save_frames(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, output_directory, frame_count=20):
    if not activate_camera_and_setup_image_memory(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch):
        return
    set_master_gain_to_zero(hCam)
    pixel_clock = set_and_get_pixel_clock(hCam, desired_pixel_clock=84)
    exposure_time = set_and_get_exposure_time(hCam, exposure_time=1.0)
    sInfo = ueye.SENSORINFO()
    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    if nRet == ueye.IS_SUCCESS:
        rectAOI = ueye.IS_RECT()
        rectAOI.s32Width = sInfo.nMaxWidth // 2
        rectAOI.s32Height = sInfo.nMaxHeight // 2
        print_camera_settings_info(sInfo, rectAOI, width, height, nBitsPerPixel)
        create_output_directory(output_directory)
        for i in range(frame_count):
            array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
            frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
            save_path = os.path.join(output_directory, f"frame_{i + 1}.png")
            cv2.imwrite(save_path, frame)
            print(f"Frame {i + 1} saved at: {save_path}")
        cleanup_camera(hCam, pcImageMemory, MemID)
        print(f"\nPixel Clock: {pixel_clock} MHz")
        print(f"Exposure Time: {exposure_time:.6f} ms")
    else:
        print("is_GetSensorInfo ERROR")

def main():
    hCam, width, height, nBitsPerPixel, pcImageMemory, MemID, bytes_per_pixel, pitch = setup_camera()
    output_directory = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\camera_outputs\\'
    capture_and_save_frames(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, output_directory, frame_count=20)

if __name__ == "__main__":
    main()
