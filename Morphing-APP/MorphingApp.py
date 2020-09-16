import sys
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QPen, QBrush, QImage
from Lab12.Morphing import *
from PIL import Image, ImageDraw, ImageQt
import os
import imageio
import numpy as np
from scipy import spatial
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene, QGraphicsItem, QGraphicsPolygonItem, QGraphicsEllipseItem
from scipy.spatial import Delaunay
from Lab12.MorphingGUI import *

class MorphingApp (QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MorphingApp, self).__init__(parent)
        self.setupUi(self)
        self.btnBlend.setEnabled(False)
        self.chkTriangles.setEnabled(False)
        self.sliderAlpha.setEnabled(False)
        self.startSetImg = QGraphicsScene()
        self.endSetImg = QGraphicsScene()
        self.Morpher = QGraphicsScene()
        self.initialPoints = False
        self.addPoints = False
        self.comfirm = False
 #       QGraphicsScene()

       # self.leftImg =
        #self.rightImg =
        self.state = "first_state"
        self.gfxLeft.setScene(self.startSetImg)
        self.gfxRight.setScene(self.endSetImg)
        self.gfxBlendImg.setScene(self.Morpher)
        self.btnStartImg.clicked.connect(self.loadLeftImage)
        self.btnEndImg.clicked.connect(self.loadRightImage)
        self.sliderAlpha.valueChanged.connect(self.setAlpha)
        self.chkTriangles.stateChanged.connect(self.triangulation)
        self.btnBlend.clicked.connect(self.getImageAtAlpha)
        self.gfxLeft.mousePressEvent = self.setLeftPoints
        self.gfxRight.mousePressEvent = self.setRightPoints
        self.centralwidget.mousePressEvent = self.secondWaySave
        self.keyPressEvent = self.BackSpace
        #self.retriangulation()




    def loadLeftImage(self):
        """
        *** DO NOT MODIFY THIS METHOD! ***
        Obtain a file name from a file dialog, and pass it on to the loading method. This is to facilitate automated
        testing. Invoke this method when clicking on the 'load' button.

        You must modify the method below.
        """
        self.filePath, _ = QFileDialog.getOpenFileName(self, caption='Open JPG file ...', filter="Image (*.jpg *.png)")

        if not self.filePath:
            return
        self.startImage = imageio.imread(self.filePath)
        self.startSetImg.clear()
        self.startSetImg.addPixmap(QPixmap(self.filePath))
        self.gfxLeft.fitInView(self.startSetImg.itemsBoundingRect() ,QtCore.Qt.KeepAspectRatio)
        self.LeftimgPoints = self.filePath + '.txt'
        try:
            fh = open(self.LeftimgPoints, 'r')
            redPen = QPen(QtCore.Qt.red)
            redBrush = QBrush(QtCore.Qt.red)
            self.leftPoints = np.loadtxt(self.LeftimgPoints)
            self.initialPoints = True
            for x, y in self.leftPoints:
                self.startSetImg.addEllipse(x,y, 20,20, redPen, redBrush)
        except FileNotFoundError:
            open(self.LeftimgPoints, 'w').close()
            self.initialPoints = False
            self.addPoints = False



    def loadRightImage(self):
        """
        *** DO NOT MODIFY THIS METHOD! ***
        Obtain a file name from a file dialog, and pass it on to the loading method. This is to facilitate automated
        testing. Invoke this method when clicking on the 'load' button.

        You must modify the method below.
        """
        self.filePath1, _ = QFileDialog.getOpenFileName(self, caption='Open JPG file ...', filter="Image (*.jpg *.png)")

        if not self.filePath1:
            return
        self.endImage = imageio.imread(self.filePath1)
        self.endSetImg.clear()
        self.endSetImg.addPixmap(QPixmap(self.filePath1))
        self.gfxRight.fitInView(self.endSetImg.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        self.btnBlend.setEnabled(True)
        self.chkTriangles.setEnabled(True)
        self.sliderAlpha.setEnabled(True)
        self.txtAlpha.setEnabled(True)

        #load point correspondence
        self.RightimgPoints = self.filePath1 + '.txt'
        try:
            fh = open(self.RightimgPoints, 'r')
            self.rightPoints = np.loadtxt(self.RightimgPoints)
            redPen = QPen(QtCore.Qt.red)
            print(self.rightPoints)
            redBrush = QBrush(QtCore.Qt.red)
            self.initialPoints = True
            for x, y in self.rightPoints:
                self.endSetImg.addEllipse(x, y, 20, 20, redPen, redBrush)
        except FileNotFoundError:
            open(self.RightimgPoints, 'w').close()
            self.initialPoints = False
            self.addPoints = False


    def setAlpha(self):
        self.txtAlpha.setText(str(self.sliderAlpha.value() / 20.0))
    def retriangulation(self):
        if self.state == "right_set":
            if self.chkTriangles.isChecked():
                self.chkTriangles.setChecked(False)
                self.chkTriangles.setChecked(True)
    def triangulation(self):
        if self.chkTriangles.isChecked() == True:
            self.tri1 = []
            self.tri2 = []
            self.leftSimplices = Delaunay(self.leftPoints).simplices
            if self.initialPoints == True and self.addPoints == True:
                Pen = QPen(QtCore.Qt.cyan)
                print(1)
            elif self.addPoints == True:
                Pen = QPen(QtCore.Qt.blue)
                print(2)
            else:
                Pen = QPen(QtCore.Qt.red)
            for triangle in self.leftSimplices.tolist():
                #TriLeft = self.leftPoints[triangle]
                #TriRight = self.rightPoints[triangle]
                #leftTriangles.append(Tri)
               # print(self.leftPoints[triangle])
                #print(self.leftPoints[triangle[0]])

                pointA = QtCore.QPointF(self.leftPoints[triangle[0]][0], self.leftPoints[triangle[0]][1])
                PointB = QtCore.QPointF(self.leftPoints[triangle[1]][0], self.leftPoints[triangle[1]][1])
                PointC = QtCore.QPointF(self.leftPoints[triangle[2]][0], self.leftPoints[triangle[2]][1])
                self.drawTriangle = QtGui.QPolygonF([pointA, PointB, PointC])
                self.tri1Item = QGraphicsPolygonItem(self.drawTriangle)
                self.tri1Item.setPen(Pen)
                self.startSetImg.addItem(self.tri1Item)


                self.tri1.append (self.tri1Item)

                pointAA = QtCore.QPointF(self.rightPoints[triangle[0]][0], self.rightPoints[triangle[0]][1])
                PointBB = QtCore.QPointF(self.rightPoints[triangle[1]][0], self.rightPoints[triangle[1]][1])
                PointCC = QtCore.QPointF(self.rightPoints[triangle[2]][0], self.rightPoints[triangle[2]][1])
                self.drawTriangle1 = QtGui.QPolygonF([pointAA, PointBB, PointCC])
                self.tri2Item = QGraphicsPolygonItem(self.drawTriangle1)
                self.tri2Item.setPen(Pen)
                self.endSetImg.addItem(self.tri2Item)
                self.tri2.append(self.tri2Item)

        else:
            #print("111111111")
            for item1 in self.tri1:
                    self.startSetImg.removeItem(item1)
            for item2 in self.tri2:
                    self.endSetImg.removeItem(item2)

    def getImageAtAlpha(self):
        triangleTuple = loadTriangles(self.filePath+'.txt',
                                      self.filePath1+'.txt')
        #img = Morpher(leftImage, triangleTuple[0], rightImage, triangleTuple[1]).getImageAtAlpha()

        alpha = self.sliderAlpha.value()
        img = Morpher(self.startImage, triangleTuple[0], self.endImage, triangleTuple[1]).getImageAtAlpha(alpha / 20.0)
        imgQt = QImage(ImageQt.ImageQt(Image.fromarray(img)))
        result = QPixmap.fromImage(imgQt)
        self.Morpher.addPixmap(result)
        self.gfxBlendImg.fitInView(self.Morpher.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
    def secondWaySave(self,e):
        if self.state == "right_set":
            self.comfirmPoints()
            self.state = "first_state"
    def setLeftPoints(self, e):
        if self.state == "first_state":
            ## if it contains points already
            self.leftPoint = self.gfxLeft.mapToScene(e.pos())
            greenPen = QPen(QtCore.Qt.green)
            greenBrush = QBrush(QtCore.Qt.green)
            self.leftPointItem = QGraphicsEllipseItem(self.leftPoint.x(), self.leftPoint.y(), 20, 20)
            self.leftPointItem.setPen(greenPen)
            self.leftPointItem.setBrush(greenBrush)
            self.startSetImg.addItem(self.leftPointItem)
            self.state = "left_set"
        elif self.state == "right_set":
            self.comfirmPoints()
            ## if it contains points already
            self.leftPoint = self.gfxLeft.mapToScene(e.pos())
            greenPen = QPen(QtCore.Qt.green)
            greenBrush = QBrush(QtCore.Qt.green)
            self.leftPointItem = QGraphicsEllipseItem(self.leftPoint.x(), self.leftPoint.y(), 20, 20)
            self.leftPointItem.setPen(greenPen)
            self.leftPointItem.setBrush(greenBrush)
            self.startSetImg.addItem(self.leftPointItem)
            self.state = "left_set"
    def setRightPoints(self, e):
        if self.state == "left_set":
            ## if it contains points already
            self.rightPoint = self.gfxRight.mapToScene(e.pos())
            greenPen = QPen(QtCore.Qt.green)
            greenBrush = QBrush(QtCore.Qt.green)
            self.rightPointItem = QGraphicsEllipseItem(self.rightPoint.x(), self.rightPoint.y(), 20, 20)
            self.rightPointItem.setPen(greenPen)
            self.rightPointItem.setBrush(greenBrush)
            self.endSetImg.addItem(self.rightPointItem)
            self.state = "right_set"
    def comfirmPoints(self):
        #print(123)
        self.addPoints = True
        self.comfirm = True
        bluePen = QPen(QtCore.Qt.blue)
        blueBrush = QBrush(QtCore.Qt.blue)
        self.startSetImg.removeItem(self.leftPointItem)
        self.endSetImg.removeItem(self.rightPointItem)
        self.leftPointItem.setPen(bluePen)
        self.leftPointItem.setBrush(blueBrush)
        self.rightPointItem.setPen(bluePen)
        self.rightPointItem.setBrush(blueBrush)
        self.endSetImg.addItem(self.rightPointItem)
        self.startSetImg.addItem(self.leftPointItem)


        if os.stat(self.LeftimgPoints).st_size == 0 and os.stat(self.RightimgPoints).st_size == 0:
            self.leftPoints = np.array([[self.leftPoint.x(), self.leftPoint.y()]])
            self.rightPoints = np.array([[self.rightPoint.x(), self.rightPoint.y()]])
        else:
            self.leftPoints = np.vstack((self.leftPoints, [self.leftPoint.x(), self.leftPoint.y()]))
            #self.leftPoints.append([self.leftPoint.x(), self.leftPoint.y()])
            #self.rightPoints.append([self.rightPoint.x(), self.rightPoint.y()])
            self.rightPoints = np.vstack((self.rightPoints, [self.rightPoint.x(), self.rightPoint.y()]))
        self.retriangulation()
        with open(self.filePath+'.txt',"w") as fin:
            for point in self.leftPoints.tolist():
                fin.write(str((round(point[0],1))) + '     ' + str((round(point[1],1))) + '\n')
        with open(self.filePath1+'.txt',"w") as fin1:
            for point in self.rightPoints.tolist():
                fin1.write(str((round(point[0],1))) + '     ' + str((round(point[1],1))) + '\n')

    def BackSpace(self, e):
        key = e.key()
        if key == QtCore.Qt.Key_Backspace:

            if self.state == "left_set":
                self.startSetImg.removeItem(self.leftPointItem)
                self.state = "first_state"
            elif self.state == "right_set":
                self.endSetImg.removeItem(self.rightPointItem)
                self.state = "left_set"

if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphingApp()

    currentForm.show()
    currentApp.exec_()