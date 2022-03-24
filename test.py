import pyrealsense2 as realsense
import numpy as np
import cv2

config = realsense.config();
config.enable_stream(realsense.stream.depth,640,480,realsense.format.z16,30);
config.enable_stream(realsense.stream.color,640,480,realsense.format.bgr8,30);
config.enable_record_to_file('objects.bag')
# Make Pipeline object to manage streaming
pipe = realsense.pipeline();


# Start streaming with default settings
profile = pipe.start(config);

e1 = cv2.getTickCount()

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        # images = (depth_colormap)
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        e2 = cv2.getTickCount()
        t = (e2 - e1) / cv2.getTickFrequency()
        if key == 27:
            cv2.destroyAllWindows()
            print("Done")
            break

finally:
    # Stop streaming
    pipe.stop()