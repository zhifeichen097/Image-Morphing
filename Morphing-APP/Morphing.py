import os
import sys
import numpy as np
import imageio
from scipy.spatial import Delaunay
#from mahotas import polygon
from PIL import Image, ImageDraw
from scipy.interpolate import RectBivariateSpline
#~ee364/DataFolder/Lab12
import matplotlib.pyplot as plt
# >>> plt.triplot(points[:,0], points[:,1], tri.simplices)
# >>> plt.plot(points[:,0], points[:,1], 'o')
def loadTriangles(leftPointFilePath, rightPointFilePath):
    #fileLeft = os.path.join(leftPointFilePath, 'TestData/LeftGray.png.txt')
    #fileRight = os.path.join(rightPointFilePath, 'TestData/RightGray.png.txt')

    leftTriangles = []
    rightTriangles = []

    #leftFile = np.loadtxt('LeftGray.png.txt')
    #print(leftFile)
    leftFile = np.loadtxt(leftPointFilePath)
    leftPoints = leftFile.astype(np.float)

    #rightFile = np.loadtxt('RightGray.png.txt')
    rightFile = np.loadtxt(rightPointFilePath)
    rightPoints = rightFile.astype(np.float)
    #print(leftPoints)
    leftSimplice = Delaunay(leftPoints).simplices
    rightSimplice = Delaunay(rightPoints).simplices
    #print(leftSimplice)
    # plt.triplot(leftPoints[:, 0], leftPoints[:, 1], leftSimplice)   ## can plot the triangles
    # plt.plot(leftPoints[:,0], leftPoints[:,1], 'o')
    # plt.show()
    #
    # plt.triplot(rightPoints[:, 0], rightPoints[:, 1], rightSimplice)  ## can plot the triangles
    # plt.plot(rightPoints[:, 0], rightPoints[:, 1], 'o')
    # plt.show()
    for triangle in leftSimplice.tolist():
        Tri = Triangle(leftPoints[triangle])
        Tri1 = Triangle(rightPoints[triangle])
        leftTriangles.append(Tri) ## 3*2 triangle vertices for left
    #for triangle in rightSimplice.tolist():
        rightTriangles.append(Tri1)  ## 3*2 triangle vertices for right
    #print(tuple((leftTriangles, rightTriangles)))
    return tuple((leftTriangles, rightTriangles))
#DataPath =

class Triangle:
    def __init__(self, vertices):
        self.vertices = vertices
        if self.vertices.dtype != 'float64':
            raise ValueError('Must be a numpy array of float64')
        if self.vertices.shape != (3,2):
            raise ValueError('Must be a 3 by 2 array')
        #if not isinstance(self.vertices, np.ndarray):
           # raise ValueError("Must be a np array")
    def getPoints(self):
        width = int(round(max(self.vertices[:,0])+2))
        height = int(round(max(self.vertices[:,1])+2))
        #print(width)
        #print(self.vertices)
        mask = Image.new('L', (width, height),0)
        ImageDraw.Draw(mask).polygon(tuple(map(tuple,self.vertices)), outline=255,fill=255)
        #print(type(self.vertices))
        #print(im)

        NbyTwo = np.transpose(np.nonzero(mask))
        #NbyTwo[:, [0,1]] = NbyTwo[:, [1,0]]

        #print((x,y))
        #print(twobyN)
        return NbyTwo
        #plt.imshow(mask)
        #plt.show()


class Morpher:
    def __init__(self, leftImage, leftTriangles, rightImage, rightTriangles):
        if not isinstance(leftImage, np.ndarray):
            raise TypeError('leftImage has to be a numpy array')
        if not isinstance(rightImage, np.ndarray):
            raise TypeError('rightImage has to be a numpy array')
        for triangle in leftTriangles:
            if not isinstance(triangle, Triangle):
                raise TypeError('leftTriangle has to be a Triangle class')
        for triangle in rightTriangles:
            if not isinstance(triangle, Triangle):
                raise TypeError('rightTriangle has to be a Triangle class')
        self.leftImage = leftImage
        self.rightImage = rightImage
        self.leftTriangles = leftTriangles
        self.rightTriangles = rightTriangles
        #self.targetImage = getTarget() #this is the target image

        #self.leftFilePath = os.path.join(os.path.expanduser('~ee364/DataFolder/Lab12/'), 'TestData/points.left.txt')
        #self.rightFilePath = os.path.join(os.path.expanduser('~ee364/DataFolder/Lab12/'), 'TestData/points.right.txt')
        #leftFile = np.loadtxt('LeftGray.png.txt')
        #leftFile = np.loadtxt(self.leftFilePath)
        #self.leftPoints = leftFile.astype(np.float)
       # rightFile = np.loadtxt('RightGray.png.txt')
        #rightFile = np.loadtxt(self.rightFilePath)
        #self.rightPoints = rightFile.astype(np.float)
        #self.leftSimplice = Delaunay(self.leftPoints).simplices
    def affineMatrix(self, original, Target):
        self.Target = Target
        self.original = original
        A = np.array([[original.vertices[0, 0], original.vertices[0, 1], 1, 0, 0, 0],
                  [0, 0, 0, original.vertices[0, 0], original.vertices[0, 1], 1],
                  [original.vertices[1, 0], original.vertices[1, 1], 1, 0, 0, 0],
                  [0, 0, 0, original.vertices[1, 0], original.vertices[1, 1], 1],
                  [original.vertices[2, 0], original.vertices[2, 1], 1, 0, 0, 0],
                  [0, 0, 0, original.vertices[2, 0], original.vertices[2, 1], 1]])
        b = np.reshape(Target.vertices, (6, 1))
        h = np.linalg.solve(A, b)
        # matrix = np.array([[h[0], h[1], h[2]],
        #                [h[3], h[4], h[5]],
        #                [0, 0, 1]
           #            ], dtype='float64')
        matrix = np.vstack([np.reshape(h, (2, 3)), [0, 0, 1]])
        self.inverseMatrix = np.linalg.inv(matrix)

    def calcXPoints(self, x, y):
        # print(np.dot(x, self.inverseMatrix[0,:]))
        return self.inverseMatrix[1, 1] * x + self.inverseMatrix[1, 0] * y + self.inverseMatrix[1, 2]
    # return self.inverseMatrix[0, 1] * x + self.inverseMatrix[0, 0] * y + self.inverseMatrix[0, 2]


    def calcYPoints(self, x, y):
        # print(np.dot(x, self.inverseMatrix[0,:]))
        return self.inverseMatrix[0, 1] * x + self.inverseMatrix[0, 0] * y + self.inverseMatrix[0, 2]
    # return self.inverseMatrix[1, 1] * x + self.inverseMatrix[1, 0] * y + self.inverseMatrix[1, 2]


    def Transformation(self, originalImage, targetImage, Target, original):  # this is image
        # print(targetimage)
        # print(xp,yp)
        #NbyTwo[:, [0, 1]] = NbyTwo[:, [1, 0]]
        NbyTwo = Target.getPoints()
        #getCoord = np.vectorize(lambda x, y, a: self.inverseMatrix[a, 0] * y + self.inverseMatrix[a, 1] * x + self.inverseMatrix[a, 2], otypes=[np.float64])
        #NbyTwo[:, [1, 0]] = NbyTwo[:, [0, 1]]
        xp, yp = np.transpose(NbyTwo)  ## this will return the points reside in one triangle, in morpher
        x = self.calcXPoints(xp, yp)
        y = self.calcYPoints(xp, yp)
        # xdim = np.array([a for a in range(int(min(original.vertices[:, 1])), int(max(original.vertices[:, 1])))])
        # # print(len(xdim))
        # ydim = np.array([a for a in range(int(min(original.vertices[:, 0])), int(max(original.vertices[:, 0])))])
        #
        # xyVal = originalImage[xdim[0]:xdim[len(xdim) - 1]+1, ydim[0]:ydim[len(ydim) - 1]+1]

        xdim = np.arange(np.amin(original.vertices[:, 1]), np.amax(original.vertices[:, 1]), 1)
        ydim = np.arange(np.amin(original.vertices[:, 0]), np.amax(original.vertices[:, 0]), 1)
        xyVal = originalImage[int(xdim[0]):int(xdim[-1] + 1), int(ydim[0]):int(ydim[-1] + 1)]
        bilinear = RectBivariateSpline(xdim, ydim, xyVal, kx=1, ky=1)
        targetImage[xp, yp] = bilinear.ev(x,y)

        # print(x)


    def getImageAtAlpha(self,alpha):
        leftTarget = np.zeros(self.leftImage.shape, dtype = 'float64') ## need to perform affine transformation to get those two
        rightTarget = np.zeros(self.rightImage.shape, dtype='float64')
        self.targetTriangles = []
        array_len = len(self.leftTriangles)
        for i in range(0, array_len):

            imageMidTargets = (1 - alpha) * self.leftTriangles[i].vertices + alpha * self.rightTriangles[i].vertices # the mid-destination for affine transformation
            self.targetTriangles.append(Triangle(imageMidTargets))
        for leftTriangle, targetTriangle in zip(self.leftTriangles, self.targetTriangles):
            self.affineMatrix(leftTriangle, targetTriangle)
            self.Transformation(self.leftImage,leftTarget, self.Target, self.original)#.Transformation(self.leftImage,leftTarget)
        #print(leftTarget)
        for rightTriangle, targetTriangle in zip(self.rightTriangles, self.targetTriangles):
            self.affineMatrix(rightTriangle, targetTriangle)
            self.Transformation(self.rightImage, rightTarget, self.Target, self.original)

        final = ((1-alpha) * leftTarget + alpha * rightTarget).astype(np.uint8)
        #print(final.astype('uint8'))
        #print(self.rightImage)
        #img = Image.fromarray(final,'L')
        #mg.show()
        return final
if __name__ == "__main__":
    leftImage = imageio.imread('~ee364/DataFolder/Lab12/TestData/LeftGray.png')
    rightImage = imageio.imread('~ee364/DataFolder/Lab12/TestData/RightGray.png')
    #print(rightImage)
    triangleTuple = loadTriangles(os.path.expanduser('~ee364/DataFolder/Lab12/'),os.path.expanduser('~ee364/DataFolder/Lab12/'))
    #print((triangleTuple[0]))
# x = np.array([1,2,3])
# x1 = 2 * x
# x2 = 3 * x
# print(x1+x2)
    Morpher(leftImage,triangleTuple[0], rightImage, triangleTuple[1]).getImageAtAlpha(0.8)
#print(triangleTuple[0][0].shape)
    #vaf = (triangleTuple[0][0].vertices)
#print(12333333333333333333333333333333333333333333333333333)
#print(triangleTuple[0])
#A = np.array([[ 1.09485880e+00 ,-1.23823316e-01, -2.57545981e+01],
 #[ 2.59232440e-01,  1.17306300e+00, -2.47030123e+02],
 #[ 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]])
#print(A[0][0])
    #print(np.arange(1440))
   # print(np.array[1:5])
#print(np.array([a for a in range(5.5, 10)]))
#A = np.array([1,2,3,4,5])



