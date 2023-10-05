#!/bin/bash

# Set the display environment variable for DLP mirror device.
export DISPLAY=:0.0

# Your video file name (replace "your_video_file.mp4" with your actual video file name).
VIDEO_FILE="binary.mp4"

# Check if the video file exists.
if [ ! -f "$VIDEO_FILE" ]; then
  echo "Video file not found: $VIDEO_FILE"
  exit 1
fi

# Run MPlayer to play the video in fullscreen mode.
mplayer -fs "$VIDEO_FILE"
