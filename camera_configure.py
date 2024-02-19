# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
from pyueye import ueye
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

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

def get_frame_rate(hCam):
    frame_rate = ueye.double()
    nRet = ueye.is_GetFramesPerSecond(hCam, frame_rate)
    if nRet != ueye.IS_SUCCESS:
        print(f"is_GetFramesPerSecond ERROR: {nRet}")
        return None
    return frame_rate.value

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

def set_and_get_aoi(hCam, sInfo):
    rectAOI = ueye.IS_RECT()
    rectAOI.s32Width = sInfo.nMaxWidth // 2
    rectAOI.s32Height = sInfo.nMaxHeight // 2
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        print("is_AOI (Set) ERROR")
        return None
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        print("is_AOI (Get) ERROR")
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
    nBitsPerPixel = ueye.INT(32)
    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
    rectAOI = set_and_get_aoi(hCam, sInfo)
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

def start_live_stream_and_save_frames(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, target_frame_count=10):
    if not activate_camera_and_setup_image_memory(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch):
        return
    frame_count = 0
    save_frames = False
    output_directory = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\camera_outputs\\'
    create_output_directory(output_directory)
    frame_counts = []
    frame_rates = []
    frames_after_enter = 0
    while True:
        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
        desired_width = 640
        desired_height = 512
        resized_frame = cv2.resize(frame, (desired_width, desired_height))
        new_frame = np.zeros((desired_height, desired_width, bytes_per_pixel), dtype=np.uint8)
        start_x = (desired_width - resized_frame.shape[1]) // 2
        start_y = (desired_height - resized_frame.shape[0]) // 2
        new_frame[start_y:start_y + resized_frame.shape[0], start_x:start_x + resized_frame.shape[1], :] = resized_frame
        cv2.imshow("Python_uEye_OpenCV", new_frame)
        frame_rate = get_frame_rate(hCam)
        frame_counts.append(frame_count)
        if save_frames:
            print(f"Frame {frames_after_enter}: {1000/frame_rate:.2f} ms")
            frame_rates.append(frame_rate)
            frames_after_enter += 1
            if frames_after_enter == target_frame_count:
                save_frames = False
                average_frame_rate = np.mean(frame_rates)
                print(f"\nAverage Frame Rate: {average_frame_rate:.2f} fps")
                average_frame_time = 1 / average_frame_rate
                print(f"Average Time per Frame: {average_frame_time * 1000:.2f} ms")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:
            save_frames = True
    plot_frame_times(frame_rates, target_frame_count)
    cv2.destroyAllWindows()

def plot_frame_times(frame_rates, frame_count):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(600)
    ax.set_xlabel("Frame Count", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time per Frame (ms)", fontsize=14, fontweight="bold")
    ax.set_title("Time per Frame vs Frame Count", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.plot(range(1, len(frame_rates) + 1), 1000 / np.array(frame_rates), marker='o', markersize=3, label='Time per Frame')
    ax.legend(loc="right", fontsize=10)
    plt.show()

def capture_single_frame(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, save_path):
    nRet = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
    if nRet == ueye.IS_SUCCESS:
        array = ueye.get_data(pcImageMemory, width, height, nBitsPerPixel, pitch, copy=False)
        frame = np.reshape(array, (height.value, width.value, bytes_per_pixel))
        frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
        cv2.destroyAllWindows()
        cv2.imwrite(save_path, frame)
    else:
        print("is_FreezeVideo ERROR")
    return save_path

def main():
    hCam, width, height, nBitsPerPixel, pcImageMemory, MemID, bytes_per_pixel, pitch = setup_camera()
    set_master_gain_to_zero(hCam)
    pixel_clock = set_and_get_pixel_clock(hCam, desired_pixel_clock = 84)
    exposure_time = set_and_get_exposure_time(hCam, exposure_time = 1.0)
    sInfo = get_sensor_info(hCam)
    rectAOI = set_and_get_aoi(hCam, sInfo)
    print_camera_settings_info(sInfo, rectAOI, width, height, nBitsPerPixel)
    output_directory = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\camera_outputs\\'
    create_output_directory(output_directory)
    print("\nPress Enter to start saving 10 frames. Press q to quit.")
    start_live_stream_and_save_frames(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, target_frame_count=10)
    cleanup_camera(hCam, pcImageMemory, MemID)
    print(f"\nPixel Clock: {pixel_clock} MHz")
    print(f"Exposure Time: {exposure_time:.6f} ms")

if __name__ == "__main__":
    main()
