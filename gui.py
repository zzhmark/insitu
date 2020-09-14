from PyQt5.QtWidgets import QDialog, QGraphicsView, QFileDialog, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi
import cv2
import os
import numpy as np
from algorithm import extract, registrate, global_gmm, local_gmm, global_score
from utils import cvimg2qpixmap, fitView

class MainDlg(QDialog):

    def __init__(self):
        super(MainDlg, self).__init__()
        loadUi('dialog.ui', self)
        self.clear()

    def enableStepBtns(self, enable):
        self.btnExtract.setEnabled(enable)
        self.btnRegistrate.setEnabled(enable)
        self.btnGlobalGmm.setEnabled(enable)
        self.btnLocalGmm.setEnabled(enable)
        self.btnHybrid.setEnabled(enable)
        self.btnScore.setEnabled(enable)
        self.btnPipeline.setEnabled(enable)

    def clear(self):
        self.tabWidget.clear()
        self.scenes = {'raw':QGraphicsScene()}
        self.rawImageView.setScene(self.scenes['raw'])
        self.images = {}
        self.data = {}
        self.enableStepBtns(False)
        self.comboBox.clear()

    @pyqtSlot()
    def on_btnOpen_clicked(self):
        """打开文件"""
        print('Loading images..')
        filenames, filter = QFileDialog.getOpenFileNames(self, "选择打开的一组图像", filter="Images (*.bmp)")
        if len(filenames) == 0:
            return
        self.clear()
        self.images['raw'] = {}
        for i in filenames:
            print(i)
            self.images['raw'][os.path.basename(i)] = cv2.imread(i, cv2.IMREAD_COLOR)
        self.enableStepBtns(True)
        self.comboBox.addItems([os.path.basename(i) for i in filenames])
        print('Done.')

    @pyqtSlot()
    def on_btnExtract_clicked(self):
        """提取"""
        if 'extract' in self.images.keys():
            print('Already extracted.')
            return
        print('Extract..')
        kernel = (3, 3)
        sd_thr = 3
        self.images['extract'] = {}
        self.data['extract'] = {}
        for key, img in self.images['raw'].items():
            self.images['extract'][key], self.data['extract'][key] = extract(img, kernel, sd_thr)
        self.newPage('extract')
        self.tabWidget.setCurrentIndex(self.tabWidget.count() - 1)
        self.on_comboBox_currentIndexChanged(0)
        print('Done.')

    @pyqtSlot(bool)
    def on_btnRegistrate_clicked(self):
        """配准"""
        if 'registrate' in self.images.keys():
            print('Already registrated.')
            return
        if not 'extract' in self.images.keys():
            self.on_btnExtract_clicked()
        print('Registrate..')
        self.images['registrate'] = {}
        self.data['registrate'] = {}
        dsize = 300, 128
        for key, img in self.images['extract'].items():
            mask = self.data['extract'][key]
            self.images['registrate'][key], self.data['registrate'][key] = registrate(img, mask, dsize)
        self.newPage('registrate')
        self.tabWidget.setCurrentIndex(self.tabWidget.count() - 1)
        self.on_comboBox_currentIndexChanged(0)
        print('Done.')

    @pyqtSlot()
    def on_btnGlobalGmm_clicked(self):
        """全局gmm"""
        if 'global gmm' in self.images.keys():
            print('Global GMM already performed.')
            return
        if not 'registrate' in self.images.keys():
            self.on_btnRegistrate_clicked()
        print('Performing Global GMM..')
        self.images['global gmm'] = {}
        self.data['global gmm mask'] = {}
        self.data['global gmm means'] = {}
        K = 10
        patch = (3, 3)
        for key, img in self.images['registrate'].items():
            mask = self.data['registrate'][key]
            image, mask, means = global_gmm(img, mask, K, patch)
            self.images['global gmm'][key] = image
            self.data['global gmm mask'][key] = mask
            self.data['global gmm means'][key] = means
        self.data['global score'] = global_score(self.data['global gmm mask'])
        self.newPage('global gmm')
        self.tabWidget.setCurrentIndex(self.tabWidget.count() - 1)
        self.on_comboBox_currentIndexChanged(0)
        print('Done.')
        print(self.data['global score'])

    @pyqtSlot()
    def on_btnLocalGmm_clicked(self):
        """局部gmm"""
        if 'local gmm' in self.images.keys():
            print('Local GMM already performed.')
            return
        if not 'global gmm' in self.images.keys():
            self.on_btnGlobalGmm_clicked()
        print('Performing Local GMM..')
        self.images['local gmm'] = {}
        self.data['local gmm'] = {}
        K = 10
        for key, img in self.images['registrate'].items():
            mask = self.data['global gmm mask'][key]
            self.images['local gmm'][key], self.data['local gmm'][key] = local_gmm(img, mask, K)
        self.newPage('local gmm')
        self.tabWidget.setCurrentIndex(self.tabWidget.count() - 1)
        self.on_comboBox_currentIndexChanged(0)
        print('Done.')

    @pyqtSlot()
    def on_btnHybrid_clicked(self):
        """融合"""
        print('hybrid')

    @pyqtSlot()
    def on_btnScore_clicked(self):
        """打分"""
        print('score')

    @pyqtSlot()
    def on_btnPipeline_clicked(self):
        """一次完成"""
        print('pipeline')

    @pyqtSlot(int)
    def on_comboBox_currentIndexChanged(self, index):
        if index == -1:
            return
        imageName = self.comboBox.currentText()
        for stepName, scene in self.scenes.items():
            scene.clear()
            scene.addPixmap(cvimg2qpixmap(fitView(self.images[stepName][imageName], scene.views()[0])))

    def newPage(self, stepName):
        self.scenes[stepName] = QGraphicsScene()
        self.tabWidget.addTab(QGraphicsView(self.scenes[stepName]), stepName)

    def removePage(self, stepName):
        n = self.tabWidget.count()
        for i in range(n):
            if self.tabWidget.tabText(i) == stepName:
                self.tabWidget.removeTab(i)
                break

