import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

cap = cv.VideoCapture("/home/ab/enpm673/project2/project2.avi")

theta_range = np.pi/180

roll_list = []
pitch_list = []
yaw_list = []

trans_x = []
trans_y = []
trans_z = []

frame_numbers = []

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv.resize(frame, (0, 0), fx= 0.5, fy= 0.5 )

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    mask = cv.inRange(gray, 200, 255)

    kernel = np.ones((3,3), np.uint8)

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    edges = cv.Canny(mask, 50, 150)

    height, width = edges.shape

    diagonal_length = int(np.sqrt(height**2 + width**2))

    hough_transform = np.zeros((2*diagonal_length, 180))

    for y in range(height):

        for x in range(width):

            if edges[y][x] != 0:

                for theta in range(0, 180):

                    d = int(x*np.cos(theta*theta_range) + y*np.sin(theta*theta_range))

                    hough_transform[d + diagonal_length][theta] += 1

    lines = []

    for i in range(4):
        max_frequency = -1
        for i in range(4):
            max_value = np.max(hough_transform)
            max_value_indices = np.argwhere(hough_transform == max_value)
            for idx in max_value_indices:
                frequency = hough_transform[idx[0], idx[1]]
                if frequency> max_frequency:
                    max_frequency = frequency

                    d = idx[0] - (diagonal_length)
                    theta = idx[1]
        lines.append((d, theta))

        max_index = np.unravel_index(np.argmax(hough_transform), hough_transform.shape)
        hough_transform[max_index[0], max_index[1]] = 0

        for j in range(-10, 10):
            for k in range(-10, 10):
                if d+j+diagonal_length >=0 and d+j+diagonal_length < 2*diagonal_length and theta+k < 180:
                    hough_transform[d+j+diagonal_length][theta+k] = 0

    for line in lines:
        d, theta = line

        a = np.cos(theta*theta_range)
        b = np.sin(theta*theta_range)

        x0 = a*d
        y0 = b*d

        x1 = int(x0 + 800*(-b))
        y1 = int(y0 + 800*(a))

        x2 = int(x0 - 800*(-b))
        y2 = int(y0 - 800*(a))

        cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    slopes = []
    intercepts = []

    for d, theta in lines:
        m = -np.cos(np.radians(theta)) / np.sin(np.radians(theta))
        b = d / np.sin(np.radians(theta))

        slopes.append(m)
        intercepts.append(b)

    intersections = []

    for i in range(len(lines)):

        for j in range(i+1, len(lines)):
            m1, b1 = slopes[i], intercepts[i]
            m2, b2 = slopes[j], intercepts[j]

            x = (b2 -b1) / (m1 - m2)
            y = m1*x + b1

            intersections.append((x, y))

    intersections.sort(key = lambda point: point[1], reverse=True)
    intersections = intersections[:4]

    for point in intersections:
        x, y = point
        if not (float('-inf') < x < float('inf') and float('-inf') < y < float('inf')):
            continue
        cv.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)

    intersections = np.array(intersections)

    paper_real = np.array([[0,0], [21.6, 0], [0, 27.9], [21.6, 27.9]])

    A = []
    for i in range(4):
        x1, y1 = paper_real[i]
        x1_dash, y1_dash = intersections[i]
        A.append([x1, y1, 1, 0, 0, 0, -x1*x1_dash, -y1*y1_dash, -x1_dash])
        A.append([0, 0, 0, x1, y1, 1, -x1*y1_dash, -y1*y1_dash, -y1_dash])  

    A = np.array(A)
    # least_product = np.dot(A.T, A)
    # eigenvalues, eigenvectors = np.linalg.eig(least_product)
    # homography = eigenvectors[:, np.argmin(eigenvalues)]
    U, S, Vt = np.linalg.svd(A)
    
    homography = Vt[-1].reshape((3, 3))

    # print(homography.shape)

    # homography = np.reshape(homography, (3,3))

    H_norm = homography / homography[2, 2]
    

    # print(H_norm)

    # print(homography)

    # print(homography)

    K = np.array([[1.38e+03/2, 0, 9.46e+02/2],
                  [0, 1.38e+03/2, 5.27e+02/2],
                  [0, 0, 1]])
    
    # To get rotation

    K_inv = np.linalg.inv(K)

    extrinsic_matrix = K_inv @ H_norm

    h1 = extrinsic_matrix[:, 0]
    h2 = extrinsic_matrix[:, 1]
    h3 = extrinsic_matrix[:, 2]

    lambdav = 1 / np.linalg.norm(h1)

    r1 = lambdav * h1
    r2 = lambdav * h2
    r3 = np.cross(r1, r2)

    rotation = np.column_stack((r1, r2, r3))

    translation = lambdav * h3

    # print(translation)
    # print(translation[0])

    trans_x.append(translation[0])
    trans_y.append(translation[1])
    trans_z.append(translation[2])

    c_matrix = np.column_stack((np.dot(K, rotation), translation))
    # print(c_matrix)

    c_pos = -np.dot(np.linalg.inv(c_matrix[:, :-1]), c_matrix[:, -1])

    R = c_matrix[:3,:3]

    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # translation = extrinsic_matrix[:, 2]

    # r1 = extrinsic_matrix[:, 0]
    # r2 = extrinsic_matrix[:, 1]

    # r1_unit = r1/np.linalg.norm(r1)
    # r2_unit = r2/np.linalg.norm(r2)

    # r3 = np.cross(r1_unit, r2_unit)

    # rotation = np.column_stack((r1_unit, r2_unit, r3))

    # rotation = Rotation.from_matrix(rotation)

    # roll, pitch, yaw = rotation.as_euler('xyz', degrees=True)
    
    roll_list.append(roll)
    pitch_list.append(pitch)
    yaw_list.append(yaw)

    cv.imshow('detected corners', frame)  
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# print(roll_list)
# print(frame_numbers)

colors = ['r'] + ['b'] * (len(roll_list) - 1)
for i in range(len(roll_list)):
    frame_numbers.append(i)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(frame_numbers, trans_x, trans_y, trans_z, c = colors)
plt.show()

plt.plot(frame_numbers, roll_list)
plt.xlabel('frames')
plt.ylabel('roll')
plt.show()

plt.plot(frame_numbers, pitch_list)
plt.xlabel('frames')
plt.ylabel('pitch')
plt.show()

plt.plot(frame_numbers, yaw_list)
plt.xlabel('frames')
plt.ylabel('yaw')
plt.show()

cap.release()
cv.destroyAllWindows()

            


