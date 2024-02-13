# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
from pyueye import ueye
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

def create_output_directory(output_directory):
    import os
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Output directory '{output_directory}' created.")

def init_camera():
    hCam = ueye.HIDS(0)
    nRet = ueye.is_InitCamera(hCam, None)
    if nRet != ueye.IS_SUCCESS:
        print("is_InitCamera ERROR")
        return None
    return hCam
def get_camera_info(hCam):
    cInfo = ueye.CAMINFO()
    nRet = ueye.is_GetCameraInfo(hCam, cInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetCameraInfo ERROR")
        return None
    return cInfo
def get_sensor_info(hCam):
    sInfo = ueye.SENSORINFO()
    nRet = ueye.is_GetSensorInfo(hCam, sInfo)
    if nRet != ueye.IS_SUCCESS:
        print("is_GetSensorInfo ERROR")
        return None
    return sInfo
def reset_camera_defaults(hCam):
    nRet = ueye.is_ResetToDefault(hCam)
    if nRet != ueye.IS_SUCCESS:
        print("is_ResetToDefault ERROR")
def set_display_mode(hCam):
    nRet = ueye.is_SetDisplayMode(hCam, ueye.IS_SET_DM_DIB)
    if nRet != ueye.IS_SUCCESS:
        print("set_display_mode ERROR:", nRet)
def set_color_mode(hCam, m_nColorMode):
    nRet = ueye.is_SetColorMode(hCam, m_nColorMode)
    if nRet != ueye.IS_SUCCESS:
        print("set_color_mode ERROR:", nRet)
def set_aoi(hCam):
    rectAOI = ueye.IS_RECT()
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        print("is_AOI ERROR")
        return None
    return rectAOI
def allocate_image_memory(hCam, width, height, nBitsPerPixel):
    pcImageMemory = ueye.c_mem_p()
    MemID = ueye.int()
    nRet = ueye.is_AllocImageMem(hCam, width, height, nBitsPerPixel, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_AllocImageMem ERROR")
        return None, None
    nRet = ueye.is_SetImageMem(hCam, pcImageMemory, MemID)
    if nRet != ueye.IS_SUCCESS:
        print("is_SetImageMem ERROR")
        return None, None
    return pcImageMemory, MemID

def setup_camera():
    hCam = init_camera()
    if hCam is None:
        return None
    cInfo = get_camera_info(hCam)
    if cInfo is None:
        ueye.is_ExitCamera(hCam)
        return None
    sInfo = get_sensor_info(hCam)
    if sInfo is None:
        ueye.is_ExitCamera(hCam)
        return None
    reset_camera_defaults(hCam)
    set_display_mode(hCam)
    m_nColorMode = ueye.INT()
    nBitsPerPixel = ueye.INT(24) # 3 channels and 8 bits per channel
    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
    rectAOI = set_aoi(hCam)
    if rectAOI is None:
        ueye.is_ExitCamera(hCam)
        return None
    width = rectAOI.s32Width
    height = rectAOI.s32Height
    pcImageMemory, MemID = allocate_image_memory(hCam, width, height, nBitsPerPixel)
    if pcImageMemory is None:
        ueye.is_ExitCamera(hCam)
        return None
    set_color_mode(hCam, m_nColorMode)
    bytes_per_pixel = int(nBitsPerPixel / 8)
    pitch = ueye.INT()
    return hCam, width, height, nBitsPerPixel, pcImageMemory, MemID, bytes_per_pixel, pitch

def activate_camera_and_setup_image_memory(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch):
    nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
    if nRet != ueye.IS_SUCCESS:
        print("is_CaptureVideo ERROR")
        return False
    nRet = ueye.is_InquireImageMem(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch)
    if nRet != ueye.IS_SUCCESS:
        print("is_InquireImageMem ERROR")
        return False
    print("Press q to leave the program")
    return True

def start_live_stream(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, duration):
    if not activate_camera_and_setup_image_memory(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch):
        return
    elapsed_time = 0
    frame_count = 0
    frame_count_start = 0 # Set to 1000 ms for experiment with DMD
    frame_times = []
    frame_counts = []
    start_time = time.perf_counter()
    while elapsed_time < duration:
        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Python_uEye_OpenCV", frame)
        frame_count += 1  
        elapsed_time = time.perf_counter() - start_time
        if frame_count >= frame_count_start:
            if frame_count > 1:
                frame_time = 1000 * elapsed_time / frame_count
                frame_times.append(frame_time)
                frame_counts.append(frame_count)
                print(f"Frame {frame_count}: {frame_time :.2f} ms per frame")
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    avg_frame_time = sum(frame_times) / len(frame_times)
    print(f"Total Number of Frames: {frame_count:} frames")
    print(f"Total Time Taken: {elapsed_time:.2f} seconds")
    print(f"Average Frame Time: {avg_frame_time:.2f} ms per frame")
    cv2.destroyAllWindows()
    plot_frame_times(frame_times, frame_count, elapsed_time, avg_frame_time)
   
def plot_frame_times(frame_times, frame_count, elapsed_time, avg_frame_time):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(600)
    ax.set_xlabel("Number of Frames", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time Taken (milliseconds)", fontsize=14, fontweight="bold")
    ax.set_title("Time Taken to Record Each Frame", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    mean_time = avg_frame_time
    median_time = sorted(frame_times)[len(frame_times) // 2]
    ax.errorbar(
        range(1, len(frame_times) + 1),
        frame_times,
        fmt='o',
        markersize=3,
        capsize=3,
        label='Time Taken',
    )
    ax.axhline(y=mean_time, color='blue', linestyle='-', label=f"Mean Time: {mean_time:.2f} ms")
    ax.axhline(y=median_time, color='green', linestyle='-', label=f"Median Time: {median_time:.2f} ms")
    ax.text(0.63, 0.80, f"Total Frames: {frame_count} frames", transform=ax.transAxes, fontsize=10, color='blue')
    ax.text(0.63, 0.76, f"Total Time Taken: {elapsed_time:.2f} seconds", transform=ax.transAxes, fontsize=10, color='blue')
    ax.legend(loc="upper right", fontsize=10)
    plt.show()
   
def capture_single_frame(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, save_path):
    nRet = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
    if nRet == ueye.IS_SUCCESS:
        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        #cv2.imshow("Captured Frame", frame)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(save_path, frame)
    else:
        print("is_FreezeVideo ERROR")
    return save_path
       
def set_exposure_time(hCam, exposure_time):
    nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.c_double(exposure_time), ueye.sizeof(ueye.c_double(exposure_time)))
    if nRet != ueye.IS_SUCCESS:
        print("is_Exposure ERROR")
       
def set_master_gain_to_zero(hCam):
    enable_auto_gain = 0.0
    ret = ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.c_double(enable_auto_gain), None)
    if ret != ueye.IS_SUCCESS:
        print("Error setting auto gain control")
    else:
        print("Auto gain control disabled")
       
def set_black_level_to_zero(hCam):
    black_level = 0.0
    nRet = ueye.is_Blacklevel(hCam, ueye.IS_BLACKLEVEL_CMD_SET_OFFSET, ueye.c_double(black_level), ueye.sizeof(ueye.c_double(black_level)))
    if nRet != ueye.IS_SUCCESS:
        print("is_Blacklevel ERROR")
    else:
        print("Black level set to zero")

def set_camera_settings(hCam):
    set_black_level_to_zero(hCam)
    set_master_gain_to_zero(hCam)
    reduced_exposure_time = 20  # in milliseconds
    set_exposure_time(hCam, reduced_exposure_time)
   
def cleanup_camera(hCam, pcImageMemory, MemID):
    ueye.is_FreeImageMem(hCam, pcImageMemory, MemID)
    ueye.is_ExitCamera(hCam)
    cv2.destroyAllWindows()

def main():
    hCam, width, height, nBitsPerPixel, pcImageMemory, MemID, bytes_per_pixel, pitch = setup_camera()
    set_camera_settings(hCam)
    output_directory = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\camera_outputs\\'
    create_output_directory(output_directory)
    frame_path =  output_directory + 'captured_frame.jpg'
    duration = 0.1
    start_live_stream(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, duration)
    capture_single_frame(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, frame_path)
    cleanup_camera(hCam, pcImageMemory, MemID)

if __name__ == "__main__":
    main()
