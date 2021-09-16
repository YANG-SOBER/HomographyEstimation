import numpy as np
import math

class Dlt(object):

  # constructor
  def __init__(self, orgs, corrs):
    self.orgs = orgs      # set of point correspondences from first image [(), (), (), ...] or [[], [], [], ...]
    self.corrs = corrs    # set of point correspondences from second image [(), (), (), ...]
  
  # compute the centroids
  def __average(self):
    sumOrgs = np.zeros(2)
    sumCorrs = np.zeros(2)
    
    for point in self.orgs:
      sumOrgs[0] += point[0] # sum of the x-coordinates of the point correspondences in the first image set
      sumOrgs[1] += point[1] # sum of the y-coordinates of ...         
    
    for point in self.corrs:
      sumCorrs[0] += point[0] # sum of the x-coordinates of the point correspondences in the second image set
      sumCorrs[1] += point[1] # sum of the y-coordinates of ...

    averageOrgs = sumOrgs / len(self.orgs)
    averageCorrs = sumCorrs / len(self.corrs)  
    
    return averageOrgs, averageCorrs
  
  # compute the scale s, which will be used in T_norm, T_norm in two images are different
  def __scale(self, points, average):
    sumOfDist = 0.0

    for point in points:
      sumOfDist += np.linalg.norm(point - average) # Frobenius norm
    
    meanOfDist = sumOfDist / len(points) # mean distance of all point correspondences from centroid (in one image)
    s = math.sqrt(2) / meanOfDist # s is scale

    return s

  # compute T_norm
  def __matrixT(self, s, average):
    Cx = average[0]
    Cy = average[1]
    T = np.float64([[s, 0, -s * Cx], 
                  [0, s, -s * Cy], 
                  [0, 0, 1]])
    return T
  
  # normalize
  def __normalize(self, points, T):
    normalizedPoints = np.zeros((len(points), 3))
    for i, point in enumerate(points):
      point = np.array([point[0], point[1], 1])
      normalizedPoints[i] = T @ point
    return normalizedPoints
  
  # Ah = 0 A: 2ix9, i is the number of point correspondences
  def __matrixA(self, normalizedOrgs, normalizedCorrs):
    A = np.zeros((len(normalizedOrgs) * 2, 9))
    
    i = 0
    for index in range(0, len(A), 2):
      A[index]= np.float64([0, 0, 0, -normalizedOrgs[i][0], -normalizedOrgs[i][1], -1, 
                        normalizedCorrs[i][1] * normalizedOrgs[i][0], normalizedCorrs[i][1] * normalizedOrgs[i][1], normalizedCorrs[i][1]])
                      
      A[index + 1] = np.float64([normalizedOrgs[i][0], normalizedOrgs[i][1], 1, 0, 0, 0, 
                           -normalizedCorrs[i][0] * normalizedOrgs[i][0], -normalizedCorrs[i][0]* normalizedOrgs[i][1], -normalizedCorrs[i][0]])
      i += 1
    return A
  
  def computeH(self):
    
    averageOrgs, averageCorrs = self.__average()
    
    sOrgs = self.__scale(self.orgs, averageOrgs)
    sCorrs = self.__scale(self.corrs, averageCorrs)
    
    T_normOrgs = self.__matrixT(sOrgs, averageOrgs)
    T_normCorrs = self.__matrixT(sCorrs, averageCorrs)
    
    normalizedOrgs = self.__normalize(self.orgs, T_normOrgs)
    normalizedCorrs = self.__normalize(self.corrs, T_normCorrs)

    A = self.__matrixA(normalizedOrgs, normalizedCorrs)

    U, S, VT = np.linalg.svd(A)
    V = VT.T
    h = V[:, -1]
    H_tilt = h.reshape(3, 3) # normalized homography
    H = np.linalg.inv(T_normCorrs) @ H_tilt @ T_normOrgs # denormalized homography

    return H 
