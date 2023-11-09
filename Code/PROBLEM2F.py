
import cv2 as cv
import numpy as np

def homography_initial(keypoints1, keypoints2):

    A = []

    for i in range(len(keypoints1)):
        x1, y1 = keypoints1[i][0][0], keypoints1[i][0][1]
        x1_dash, y1_dash = keypoints2[i][0][0], keypoints2[i][0][1]
        A.append([x1, y1, 1, 0, 0, 0, -x1*x1_dash, -y1*y1_dash, -x1_dash])
        A.append([0, 0, 0, x1, y1, 1, -x1*y1_dash, -y1*y1_dash, -y1_dash]) 

    A = np.asarray(A)

    eigenvalue, eigenvector = np.linalg.eig(A.T @ A)

    h = eigenvector[:, np.argmin(eigenvalue)]

    homography = h.reshape((3, 3))

    return homography

def perspective_transform(transformed_points, homog):
    src_hom = np.concatenate(
        (transformed_points, np.ones((len(transformed_points), 1, 1), dtype=np.float32)), axis=2)

    new_homogeneous_points = np.matmul(src_hom, homog.T)

    euc_coordinates = new_homogeneous_points[:, :, :2] / new_homogeneous_points[:, :, 2:]

    return euc_coordinates

def ransac1(i1, i2):

    gray1 = cv.cvtColor(i1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(i2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)

    # img1_extract = cv.drawKeypoints(img1, keypoints1, None)

    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # img2_extract = cv.drawKeypoints(img2, keypoints2, None)

    flann_matcher = cv.FlannBasedMatcher_create()

    match_12 = flann_matcher.knnMatch(descriptors1, descriptors2, k=2)

    best_match_12 = []
    for i, j in match_12:
        if i.distance < 0.25 * j.distance:
            best_match_12.append(i)

    # draw_match12 = cv.drawMatches(img1, keypoints1, img2, keypoints2, best_match_12, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in best_match_12 ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in best_match_12 ]).reshape(-1, 1, 2)
    
    threshold = 10
    iterations = 1000

    best_homography = None
    num_inliners = 0

    for i in range(iterations):
        rand_index = np.random.choice(len(src_pts), 4, replace= False)
        sample1 = src_pts[rand_index]
        sample2 = dst_pts[rand_index]

        new_homography = homography_initial(sample2, sample1)
        new_points = perspective_transform(dst_pts, new_homography)

        distance = np.linalg.norm(src_pts - new_points, axis = 2)
        inliners = np.sum(distance<threshold)

        if inliners > num_inliners:
            num_inliners = inliners
            best_homography = new_homography
    
        
    final_image = cv.warpPerspective(i2, best_homography, ((2*i1.shape[1] + i2.shape[1]), (i1.shape[0])))
    final_image[0:i1.shape[0], 0:i1.shape[1]] = i1

    return final_image


def ransac(i1, i2):

    gray1 = cv.cvtColor(i1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(i2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)

    # img1_extract = cv.drawKeypoints(img1, keypoints1, None)

    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # img2_extract = cv.drawKeypoints(img2, keypoints2, None)

    flann_matcher = cv.FlannBasedMatcher_create()

    match_12 = flann_matcher.knnMatch(descriptors1, descriptors2, k=2)

    best_match_12 = []
    for i, j in match_12:
        if i.distance < 0.25 * j.distance:
            best_match_12.append(i)

    # draw_match12 = cv.drawMatches(img1, keypoints1, img2, keypoints2, best_match_12, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in best_match_12 ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in best_match_12 ]).reshape(-1, 1, 2)
    
    threshold = 10
    iterations = 1000

    best_homography = None
    num_inliners = 0

    for i in range(iterations):
        rand_index = np.random.choice(len(src_pts), 4, replace= False)
        sample1 = src_pts[rand_index]
        sample2 = dst_pts[rand_index]

        new_homography = homography_initial(sample2, sample1)
        new_points = perspective_transform(dst_pts, new_homography)

        distance = np.linalg.norm(src_pts - new_points, axis = 2)
        inliners = np.sum(distance<threshold)

        if inliners > num_inliners:
            num_inliners = inliners
            best_homography = new_homography
    
    
    final_image = cv.warpPerspective(i2, best_homography, ((i1.shape[1] + i2.shape[1]), (i1.shape[0])))
    final_image[0:i1.shape[0], 0:i1.shape[1]] = i1

    return final_image

img1 = cv.imread('/home/ab/enpm673/project2/question 2/image_1.jpg')
img2 = cv.imread('/home/ab/enpm673/project2/question 2/image_2.jpg')
img3 = cv.imread('/home/ab/enpm673/project2/question 2/image_3.jpg')
img4 = cv.imread('/home/ab/enpm673/project2/question 2/image_4.jpg')


#stitching from 1 to 4

stiched_12 = ransac1(img1, img2)

white_pixels = np.any(stiched_12 != [0, 0 , 0 ], axis= -1)
y_min, x_min = np.min(np.where(white_pixels), axis = 1)
y_max, x_max = np.max(np.where(white_pixels), axis = 1)

cropped_image1 = stiched_12[y_min:y_max+1, x_min:x_max+1]

stiched_34 = ransac1(img3, img4)

white_pixels = np.any(stiched_34 != [0, 0 , 0 ], axis= -1)
y_min, x_min = np.min(np.where(white_pixels), axis = 1)
y_max, x_max = np.max(np.where(white_pixels), axis = 1)

cropped_image2 = stiched_34[y_min:y_max+1, x_min:x_max+1]

stiched_1234 = ransac1(cropped_image1, cropped_image2)

cv.namedWindow('stitched image', cv.WINDOW_NORMAL)
cv.resizeWindow('stitched image', (img2.shape[1]+img2.shape[1]), (img2.shape[0] + img2.shape[0]))

cv.imshow('stitched image', stiched_1234)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif cv.getWindowProperty('stitched image', cv.WND_PROP_VISIBLE) < 1:
        break
cv.destroyAllWindows()



# Stitching from 4 to 1

stiched_34_reverse = ransac(img3, img4)

stiched_234_reverse = ransac(img2, stiched_34_reverse)

stiched_1234_reverse = ransac(img1, stiched_234_reverse)


cv.namedWindow('stitched image inverse', cv.WINDOW_NORMAL)
cv.resizeWindow('stitched image inverse', (img2.shape[1]+img2.shape[1]), (img2.shape[0] + img2.shape[0]))

cv.imshow('stitched image inverse', stiched_1234_reverse)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif cv.getWindowProperty('stitched image inverse', cv.WND_PROP_VISIBLE) < 1:
        break
cv.destroyAllWindows()

