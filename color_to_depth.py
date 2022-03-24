"""
Author: Konstantinos Angelopoulos
Date: 04/02/2020
All rights reserved.
Feel free to use and modify and if you like it give it a star.
"""
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np

# Retrieve the depth camera intrinsics from the kinect's mapper
# and write them at: calibrate/IR/intrinsics_retrieved_from_kinect_mapper.json
def intrinsics(kinect, path='calibrate/IR/intrinsics_retrieved_from_kinect_mapper.json', write=False):
    """
    :param kinect: kinect instance
    :param path: path to save the intrinsics as a json file
    :param write: save or not save the intrinsics
    :return: returns the intrinsics matrix
    """
    import json
    intrinsics_matrix = kinect._mapper.GetDepthCameraIntrinsics()
    if write:
        with open(path, 'w', encoding='utf-8') as json_file:
            configs = {"FocalLengthX": intrinsics_matrix.FocalLengthX, "FocalLengthY": intrinsics_matrix.FocalLengthY,
                       "PrincipalPointX": intrinsics_matrix.PrincipalPointX, "PrincipalPointY": intrinsics_matrix.PrincipalPointY,
                       "RadialDistortionFourthOrder": intrinsics_matrix.RadialDistortionFourthOrder, "RadialDistortionSecondOrder": intrinsics_matrix.RadialDistortionSecondOrder,
                       "RadialDistortionSixthOrder": intrinsics_matrix.RadialDistortionSixthOrder}
            json.dump(configs, json_file, separators=(',', ':'), sort_keys=True, indent=4)
    return intrinsics_matrix




if __name__ == '__main__':
    global transform
    """
        Example of some usages
    """
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
    transform = np.array([[0.3584, 0.0089, 0.0000],
        [0.0031, 0.3531, 0.0001],
        [-101.5934, 13.6311, 0.9914]])

    pixels = np.zeros((1920,1080,2))
    for x,y in zip(range(1,1920), range(1,1080)):
        depth_pixels = np.dot(transform,[x,y,1])
        depth_pixels = depth_pixels/depth_pixels[-1]
        pixels[x,y] = depth_pixels[0:2]

    for x,y in pixels.reshape(-1,2):
        count = 0
        if x>0 and y>0:
            print(x,y)
            count += 1
    print("FINAL COUNT",count)

    # print(pixels[500,500])
    while True:
        if kinect.has_new_depth_frame():
            color_frame = kinect.get_last_color_frame()
            colorImage = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(np.uint8)
            colorImage = cv2.flip(colorImage, 1)
            cv2.imshow('Test Color View', cv2.resize(colorImage, (int(1920 / 2.5), int(1080 / 2.5))))
            depth_frame = kinect.get_last_depth_frame()
            depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)).astype(np.uint8)
            depth_img = cv2.flip(depth_img, 1)
            cv2.imshow('Test Depth View', depth_img)
            intrinsics_matrix = intrinsics(kinect)
            # print(intrinsics_matrix)
        # Quit using q
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
