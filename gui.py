from PyQt5.QtWidgets import QDialog, QGraphicsView, QFileDialog, QGraphicsScene
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi
import cv2
import os
from algorithm import batch_apply
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
        self.masks = {}
        self.scores = {}
        self.means = {}
        self.enableStepBtns(False)
        self.comboBox.clear()

    @pyqtSlot()
    def on_btnOpen_clicked(self):
        """打开文件"""
        print('Loading images..')
        filenames, filter = QFileDialog.getOpenFileNames(self,
                                                         "选择打开的一组图像",
                                                         filter="Images (*.bmp)")
        if len(filenames) == 0:
            return
        self.clear()
        basenames = [os.path.basename(i) for i in filenames]
        img_dict = dict.fromkeys(basenames)
        for full, base in zip(filenames, basenames):
            print(full)
            img_dict[base] = cv2.imread(full, cv2.IMREAD_COLOR)
        self.images['raw'] = img_dict
        self.enableStepBtns(True)
        self.comboBox.addItems(basenames)
        print('Done.')

    @pyqtSlot()
    def on_btnExtract_clicked(self):
        step = 'extract'
        if step in self.images.keys():
            print('Already extracted.')
            return
        print('Extracting..')

        self.images[step], self.masks[step] = \
            batch_apply(step,
                        images = self.images['raw'],
                        kernel = (3, 3),
                        sd_thr = 3)

        self.newPage(step)
        self.tabWidget.setCurrentIndex(self.tabWidget.count() - 1)
        self.on_comboBox_currentIndexChanged(0)
        print('Done.')

    @pyqtSlot(bool)
    def on_btnRegistrate_clicked(self):
        step = 'registrate'
        if step in self.images.keys():
            print('Already registrated.')
            return
        if not 'extract' in self.images.keys():
            self.on_btnExtract_clicked()
        print('Registrating..')

        self.images[step], self.masks[step] = \
            batch_apply(step,
                        keys = self.images['raw'].keys(),
                        images = self.images['extract'].values(),
                        masks = self.masks['extract'].values(),
                        dsize = (300, 120))

        self.newPage(step)
        self.tabWidget.setCurrentIndex(self.tabWidget.count() - 1)
        self.on_comboBox_currentIndexChanged(0)
        print('Done.')

    @pyqtSlot()
    def on_btnGlobalGmm_clicked(self):
        step = 'global gmm'
        if step in self.images.keys():
            print('Global GMM already performed.')
            return
        if not 'registrate' in self.images.keys():
            self.on_btnRegistrate_clicked()
        print('Performing Global GMM..')

        self.images[step], self.masks[step], self.means[step], self.scores[step] = \
            batch_apply(step,
                        keys = self.images['raw'].keys(),
                        images = self.images['registrate'].values(),
                        masks = self.masks['registrate'].values(),
                        K = 5,
                        patch = (3, 3))

        self.newPage('global gmm')
        self.tabWidget.setCurrentIndex(self.tabWidget.count() - 1)
        self.on_comboBox_currentIndexChanged(0)
        print('Done.')
        print(self.scores[step])

    @pyqtSlot()
    def on_btnLocalGmm_clicked(self):
        step = 'local gmm'
        if step in self.images.keys():
            print('Local GMM already performed.')
            return
        if not 'global gmm' in self.images.keys():
            self.on_btnGlobalGmm_clicked()
        print('Performing Local GMM..')

        self.images[step], self.masks[step], self.means[step], self.scores[step] = \
            batch_apply(step,
                        keys = self.images['raw'].keys(),
                        images = self.images['registrate'].values(),
                        masks = self.masks['global gmm'].values(),
                        means = self.means['global gmm'].values(),
                        K = 10)

        self.newPage(step)
        self.tabWidget.setCurrentIndex(self.tabWidget.count() - 1)
        self.on_comboBox_currentIndexChanged(0)
        print('Done.')
        print(self.scores[step])

    @pyqtSlot()
    def on_btnHybrid_clicked(self):
        step = 'hybrid'
        if step in self.images.keys():
            print('Hybriding already performed.')
            return
        if not 'local gmm' in self.images.keys():
            self.on_btnLocalGmm_clicked()
        print('Hybriding..')


    @pyqtSlot()
    def on_btnScore_clicked(self):
        """打分"""
        print('score')

    @pyqtSlot()
    def on_btnExport_clicked(self):
        """一次完成"""
        print('export')

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

