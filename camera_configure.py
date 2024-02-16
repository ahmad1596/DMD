# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
from pyueye import ueye
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import os

def create_output_directory(output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Output directory '{output_directory}' created.")

def get_pixel_clock(hCam):
    pixel_clock = ueye.c_uint()
    nRet = ueye.is_PixelClock(hCam, ueye.IS_PIXELCLOCK_CMD_GET, pixel_clock, ueye.sizeof(pixel_clock))
    if nRet != ueye.IS_SUCCESS:
        print("is_PixelClock ERROR")
        return None
    return pixel_clock.value

def get_frame_rate(hCam):
    frame_rate = ueye.double()
    nRet = ueye.is_GetFramesPerSecond(hCam, frame_rate)
    if nRet != ueye.IS_SUCCESS:
        print(f"is_GetFramesPerSecond ERROR: {nRet}")
        return None
    return frame_rate.value

def get_exposure_time(hCam):
    exposure_time = ueye.c_double()
    nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, exposure_time, ueye.sizeof(exposure_time))
    if nRet != ueye.IS_SUCCESS:
        print("is_Exposure ERROR")
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

def set_aoi(hCam, sInfo):
    rectAOI = ueye.IS_RECT()
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
    if nRet != ueye.IS_SUCCESS:
        print("is_AOI ERROR")
        return None
    rectAOI.s32Width = sInfo.nMaxWidth // 2
    rectAOI.s32Height = sInfo.nMaxHeight // 2
    nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rectAOI, ueye.sizeof(rectAOI))
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
    nBitsPerPixel = ueye.INT(24)
    if int.from_bytes(sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
        ueye.is_GetColorDepth(hCam, nBitsPerPixel, m_nColorMode)
    rectAOI = set_aoi(hCam, sInfo)
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

def set_exposure_time(hCam, exposure_time):
    nRet = ueye.is_Exposure(hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, ueye.c_double(exposure_time), ueye.sizeof(ueye.c_double(exposure_time)))
    if nRet != ueye.IS_SUCCESS:
        print("is_Exposure ERROR")

def set_master_gain_to_zero(hCam):
    enable_auto_gain = 0.0
    ret = ueye.is_SetAutoParameter(hCam, ueye.IS_SET_ENABLE_AUTO_GAIN, ueye.c_double(enable_auto_gain), None)
    if ret != ueye.IS_SUCCESS:
        print("Error setting auto gain control")

def set_camera_settings(hCam):
    set_master_gain_to_zero(hCam)
    reduced_exposure_time = 1  # in milliseconds
    set_exposure_time(hCam, reduced_exposure_time)

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
    elapsed_time = 0
    frame_count = 0
    frame_times = []
    start_time = 0
    save_frames = False
    output_directory = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\camera_outputs\\'
    create_output_directory(output_directory)
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
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:
            save_frames = True
            start_time = time.perf_counter()
        if save_frames and frame_count < target_frame_count:
            frame_count += 1
            elapsed_time = time.perf_counter() - start_time
            frame_time = 1000 * elapsed_time / frame_count
            frame_times.append(frame_time)
            print(f"Frame {frame_count}: {frame_time:.2f} ms per frame")
            frame_path = os.path.join(output_directory, f'captured_frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, new_frame)  # Save the new frame instead of the original
        if frame_count == target_frame_count:
            save_frames = False
    avg_frame_time = sum(frame_times) / len(frame_times)
    print(f"\nTotal Number of Frames: {frame_count} frames")
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
    median_time = sorted(frame_times)[len(frame_times)

 // 2]
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
    ax.legend(loc="lower right", fontsize=10)
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
    set_camera_settings(hCam)
    pixel_clock = get_pixel_clock(hCam)
    frame_rate = get_frame_rate(hCam)
    exposure_time = get_exposure_time(hCam)
    sInfo = get_sensor_info(hCam)
    rectAOI = set_aoi(hCam, sInfo)
    print_camera_settings_info(sInfo, rectAOI, width, height, nBitsPerPixel)
    output_directory = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\camera_outputs\\'
    create_output_directory(output_directory)
    print("\nPress Enter to start saving 10 frames. Press q to quit.")
    start_live_stream_and_save_frames(hCam, pcImageMemory, MemID, width, height, nBitsPerPixel, pitch, bytes_per_pixel, target_frame_count=10)
    cleanup_camera(hCam, pcImageMemory, MemID)
    print(f"\nPixel Clock: {pixel_clock} MHz")
    print(f"Frame Rate: {frame_rate} fps")
    print(f"Exposure Time: {exposure_time:.6f} ms")
    
if __name__ == "__main__":
    main()
