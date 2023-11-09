# Camera-Pose-Estimation-Using-Homography

## Libraries Required

The project requires the following libraries. Please ensure they are installed before running the code:

- `opencv`: For computer vision tasks, including video processing and feature detection.
- `numpy`: For numerical computations and matrix operations.
- `matplotlib.pyplot`: For plotting graphs and visualizations.
- `mpl_toolkits.mplot3d`: For 3D plotting capabilities.
- `sift`: For extracting and locating features in the images.
- `flannbasedmatcher`: For efficiently matching the SIFT features between images.

## Problem 1: Video Path Configuration and Pose Estimation

### Video Path Configuration

Before executing the code, update the path to the video file according to your local file system setup.

### Execution Details

- The code processes a total of 146 frames from the video, which may take some time to run depending on system performance.
- After processing the video, the code will display a series of four plots:
  1. Translation plot showing the movement of the camera.
  2. Roll plot indicating the rotation around the front-to-back axis.
  3. Pitch plot illustrating the rotation around the side-to-side axis.
  4. Yaw plot depicting the rotation around the vertical axis.
  
These plots are with respect to the frames processed from the video.

### Additional Information

- There is no need to install any non-standard libraries beyond those listed above.
- Ensure that all paths and environment settings are correct for your operating system.

![Initial Plot](Results\corner_detection.png)
![Fitted Plot](Results\edges.png)
![Initial Plot](Results\translation.png)

<img src ="Results\corner_detection.png" width=400/>
<img src ="Results\edges.png" width=400/>
