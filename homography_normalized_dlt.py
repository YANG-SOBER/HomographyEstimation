import cv2
import numpy as np
import random
import sys
import getopt
from Normalized_DLT import *


# Read in an image file, errors out if we can't find the file
def readImage(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image' + filename)
    else:
        print('Image successfully read...')

    return img

# This draws matches and optionally a set of inliers in a different color
def drawMatches(img1, kp1, img2, kp2, matches, inliers=None):
    row1 = img1.shape[0]
    row2 = img2.shape[0]
    col1 = img1.shape[1]
    col2 = img2.shape[1]

    out = np.zeros((max(row1, row2), col1 + col2, 3), dtype='uint8')

    out[:row1, :col1, :] = np.dstack((img1, img1, img1))

    out[:row2, col1:col1+col2, :] = np.dstack((img2, img2, img2))

    for match in matches:
        img1_idx = match.queryIdx # index of the descriptor in query descriptors
        img2_idx = match.trainIdx # index of the descriptor in train descriptors

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True
        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+col1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+col1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+col1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+col1,int(y2)), (255, 0, 0), 1)

    return out

# Run SIFT algorithm to find features
def findFeatures(img):
    # BGR to Gray
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Compute keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None) # List of keypoints, Numpy array of descriptors (Number of Keypoints, 128)

    # Draw keypoints
    img_kps = cv2.drawKeypoints(img, keypoints, outImage=None)

    # Store the image with keypoints
    cv2.imwrite("Image_with_SIFT_Keypoints.jpg", img_kps)

    return keypoints, descriptors

# Matches features given a list of keypoints, descriptors, and images
def matchFeatures(kp1, kp2, des1, des2, img1, img2):
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Draw matches
    img_match = drawMatches(img1, kp1, img2, kp2, matches)

    # Store the image with matches
    cv2.imwrite("Image_with_BFMatcher_Keypoints.jpg", img_match)

    return matches

# Compute a homography from 4-correspondences

def calculateHomography(correspondences):
    A_list = []
    for corr in correspondences: # corr: np.array([x1, y1, x2, y2])

        # write each point correspondence as homogeneous coordinates
        p1 = np.array([corr.item(0), corr.item(1), 1]) # [x1, y1, 1]
        p2 = np.array([corr.item(2), corr.item(3), 1]) # [x2, y2, 1]

        # Construct Ai, s.t. Aih = 0, where Ai is (2, 9), h is (9, 1)
        # A_1 = [0, 0, 0, -x_1, -y_1, -1, y_2*x_1, y_2*y_1, y_2]
        A_1 = [0, 0, 0, -p1.item(0), -p1.item(1), -1, p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1)]
        # A_2 = [x1, y1, 1, 0, 0, 0, -x_2*x_1, -x_2*y_1, -x_2]
        A_2 = [p1.item(0), p1.item(1), 1, 0, 0, 0, -p2.item(0) * p1.item(0), -p2.item(0) * p1.item(1), -p2.item(0)]

        A_list.append(A_1)
        A_list.append(A_2)
    # Construct A, s.t. Ah = 0, where A is (2* len(correspondences), 9)
    A = np.array(A_list)

    # compute and transform h, to get homography matrix H
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    h = V[:, -1] # h is in the shape of (9, 1)
    h /= h.item(8)
    H = h.reshape((3, 3)) # or H = np.reshape(h, (3, 3))

    return H

# Calculate the geometric distance between estimated points and original points
def geometricDistance(correspondence, H):
    p1 = np.array([correspondence.item(0), correspondence.item(1), 1]) # p1: [x1, y1, 1] homogeneous coordinates
    p2_esti = H @ p1 # p2 shape: (3,)

    p2 = np.array([correspondence.item(2), correspondence.item(3), 1]) # p2: [x2, y2, 1] homogeneous coordinates
    error = p2_esti - p2
    geo_dist = np.linalg.norm(error)
    return geo_dist

# Run through ransac algorithm, create homographies from correspondences
def ransac(corr, thresh): # thresh: inliers ratio threshold

    dist_thresh = 5 # 5 pix
    maxInliers = []
    pt1_list = [] # [[x1, y1], [], [], ...]
    pt2_list = [] # [[x2, y2], [], [], ...]

    # Repeat for 1000 trials
    for trial in range(1000):
        # *Sample* Select a random sample of 4 correspondences

        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        corrFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        corrFour = np.vstack((corrFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        corrFour = np.vstack((corrFour, corr4))

        for i in range(len(corrFour)):
            pt1_list.append(list(corrFour[:, :2][i]))
            pt2_list.append(list(corrFour[:, 2:][i]))

        #with open("point_correspondences.txt", 'w') as f:
            #f.write(f'4 point correspondences: \n {corrFour} \n')
            #f.write(f'pt1: \n {pt1_list} \n')
            #f.write(f'pt2: \n {pt2_list} \n')

        # *Compute* Compute the homography from the above 4 randomly sampled points
        # H = calculateHomography(corrFour)
        normalized_dlt = Dlt(pt1_list, pt2_list)
        H = normalized_dlt.computeH()
        H /= H.item(-1)
        inliers = []

        # *Calculate* Calculate the distance d for each putative correspondence
        for i in range(len(corr)):
            d = geometricDistance(corr[i], H)
            # Compute the number of inliers consistent with H by the number of correspondences for which d < dist_thresh
            if d < dist_thresh:
                inliers.append(corr[i]) # [array1, array2, ...]

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = H
        print(f"Corr size: {len(corr)}, Number of Inliers: {len(inliers)}, Number of maxInliers: {len(maxInliers)}")

        if len(maxInliers) > (len(corr) * thresh):
            break
    return finalH, maxInliers

def main():
    opts, args = getopt.getopt(sys.argv[1:], '', ['threshold=']) # opts: [('threshold', '0.6')] args: ['img1.png', 'img2.png']
    opts_dict = dict(opts)

    # get the thresh from the command line
    thresh = float(opts_dict.get('--threshold'))

    # get the image path from the command line
    img1_path = args[0]
    img2_path = args[1]

    # read the two images
    img1 = readImage(img1_path)
    img2 = readImage(img2_path)

    # find keypoints and descriptors for two images
    kp1, des1 = findFeatures(img1)
    kp2, des2 = findFeatures(img2)

    # Match features
    matches = matchFeatures(kp1, kp2, des1, des2, img1, img2)

    # Matrix of point correspondences
    '''
    np.array([[x1, y1, x2, y2],
              [x1, y1, x2, y2],
              ...])
    '''
    corrs_list = []
    for match in matches:
        img1_idx = match.queryIdx # index of descriptor in query descriptors
        img2_idx = match.trainIdx # index of descriptor in train descriptors

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        corr = [x1, y1, x2, y2]
        corrs_list.append(corr)

    corrs = np.array(corrs_list)

    # Run RANdom SAmple Consensus algorithm
    finalH, inliers = ransac(corrs, thresh)
    print(f"Final Homography: {finalH}")
    print(f"Final inliers count: ", len(inliers))

    matchImg = drawMatches(img1, kp1, img2, kp2, matches, inliers)
    cv2.imwrite("InlierMatches.png", matchImg)

    with open("homography.txt", 'w') as f:
        f.write(f"Final homography: \n {finalH} \n")
        f.write(f"Final inliers count: {len(inliers)}")

if __name__ == '__main__':
    main()
