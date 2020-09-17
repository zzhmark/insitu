from PyQt5.QtWidgets import QDialog, QFileDialog, QGraphicsScene
from PyQt5.QtCore import pyqtSlot, QModelIndex
from PyQt5.uic import loadUi
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import cv2
import os
from algorithm import batch_apply
from utils import cvimg2qpixmap, fitView, Step
import pandas as pd

class MainDlg(QDialog):

    def __init__(self):
        super(MainDlg, self).__init__()
        loadUi('dialog.ui', self)
        self.scenes = {i:QGraphicsScene() for i in Step}
        self.viewRaw.setScene(self.scenes[Step.RAW])
        self.viewExtract.setScene(self.scenes[Step.EXTRACT])
        self.viewRegister.setScene(self.scenes[Step.REGISTER])
        self.viewGlobalGmm.setScene(self.scenes[Step.GLOBAL_GMM])
        self.viewLocalGmm.setScene(self.scenes[Step.LOCAL_GMM])
        self.selectorModel = QStandardItemModel()
        self.viewSelector.setModel(self.selectorModel)
        self.reset()
        self.supportedFormats = ['.bmp']
        self.viewSelector.setColumnWidths([178] * 4)

    def enableStepButtons(self, enable):
        self.btnExtract.setEnabled(enable)
        self.btnRegister.setEnabled(enable)
        self.btnGlobalGmm.setEnabled(enable)
        self.btnLocalGmm.setEnabled(enable)
        self.btnScore.setEnabled(enable)
        self.btnExport.setEnabled(enable)

    def reset(self):
        self.images = {Step.RAW: {}}
        self.masks = {}
        self.scores = {}
        self.mean_sets = {}
        self.metadata = pd.DataFrame()
        self.selectorModel.clear()
        for scene in self.scenes.values():
            scene.clear()
        self.enableStepButtons(False)

    def undoSince(self, step):
        for data in [self.images, self.masks, self.mean_sets, self.scores]:
            keys = [key for key in data.keys() if key.value >= step.value]
            [*map(data.pop, keys)]

    def updateImages(self, index):
        if index == QModelIndex():
            return
        id = self.selectorModel.itemFromIndex(index).text()
        for scene in self.scenes.values():
            scene.clear()
        for step in self.images.keys():
            img_resize = fitView(self.images[step][id], self.scenes[step].views()[0])
            self.scenes[step].addPixmap(cvimg2qpixmap(img_resize))

    @pyqtSlot()
    def on_btnImport_clicked(self):
        print('Importing images..')
        step = Step.RAW
        dir = QFileDialog.getExistingDirectory(self, '选择一个目录', '', QFileDialog.ShowDirsOnly)
        if dir == '':
            return

        self.reset()
        self.images[step] = {}
        for gene in os.listdir(dir):
            gene_dir = os.path.join(dir, gene)
            if not os.path.isdir(gene_dir):
                continue
            gene_item = QStandardItem(gene)
            self.selectorModel.appendRow(gene_item)
            for stage in os.listdir(gene_dir):
                stage_dir = os.path.join(gene_dir, stage)
                if not os.path.isdir(stage_dir):
                    continue
                stage_item = QStandardItem(stage)
                gene_item.appendRow(stage_item)
                for file in os.listdir(stage_dir):
                    id, ext = os.path.splitext(file)
                    if ext not in self.supportedFormats:
                        continue
                    filename = os.path.join(dir, gene, stage, file)
                    self.images[step][id] = cv2.imread(filename, cv2.IMREAD_COLOR)
                    stage_item.appendRow(QStandardItem(id))
                    self.metadata = self.metadata.append([[gene, stage, id]])
        if self.images[step] == {}:
            self.reset()
            return

        self.metadata.columns = ['gene', 'stage', 'id']
        self.metadata = self.metadata.set_index('id')
        self.enableStepButtons(True)
        print('Done.')

    @pyqtSlot()
    def on_btnExtract_clicked(self):
        sd_kernel_size_input = self.edtSdKernelSize.text()
        sd_thr_input = self.edtSdThr.text()
        if sd_kernel_size_input.isnumeric() and sd_thr_input.isnumeric():
            sd_kernel = (int(sd_kernel_size_input),) * 2
            sd_thr = int(sd_thr_input)
        else:
            print('Invalid input!')
            return
        print('Extracting..')
        step = Step.EXTRACT
        self.undoSince(step)
        self.images[step], self.masks[step] = \
            batch_apply(step, 
                        images = self.images[Step.RAW], 
                        kernel = sd_kernel,
                        thr = sd_thr)
        self.updateImages(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot(bool)
    def on_btnRegister_clicked(self):
        width_input = self.edtWidth.text()
        height_input = self.edtHeight.text()
        if width_input.isnumeric() and height_input.isnumeric():
            size = (int(width_input), int(height_input))
        else:
            print('Invalid input!')
            return
        if Step.EXTRACT not in self.images.keys():
            self.on_btnExtract_clicked()
        print('Registering..')
        step = Step.REGISTER
        self.undoSince(step)
        self.images[step], self.masks[step] = \
            batch_apply(step,
                        keys = self.metadata.index,
                        images = self.images[Step.EXTRACT].values(),
                        masks = self.masks[Step.EXTRACT].values(),
                        size = size)
        self.updateImages(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot()
    def on_btnGlobalGmm_clicked(self):
        num_kernel_input = self.edtGlobalNumKernel.text()
        pooling_rate_input = self.edtPoolingRate.text()
        if num_kernel_input.isnumeric() and pooling_rate_input.isnumeric():
            num_kernel = int(num_kernel_input)
            patch = (int(pooling_rate_input),) * 2
        else:
            print('Invalid input!')
            return
        if Step.REGISTER not in self.images.keys():
            self.on_btnRegister_clicked()
        print('Performing Global GMM..')
        step = Step.GLOBAL_GMM
        self.undoSince(step)
        self.images[step], self.masks[step], self.mean_sets[step] = \
            batch_apply(step,
                        keys = self.metadata.index,
                        images = self.images[Step.REGISTER].values(),
                        masks = self.masks[Step.REGISTER].values(),
                        nok = num_kernel,
                        patch = patch)
        self.updateImages(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot()
    def on_btnLocalGmm_clicked(self):
        num_kernel_input = self.edtGlobalNumKernel.text()
        if num_kernel_input.isnumeric():
            num_kernel = int(num_kernel_input)
        else:
            print('Invalid input!')
            return
        if Step.GLOBAL_GMM not in self.images.keys():
            self.on_btnGlobalGmm_clicked()
        print('Performing Local GMM..')
        step = Step.LOCAL_GMM
        self.undoSince(step)
        self.images[step], self.masks[step], self.mean_sets[step] = \
            batch_apply(step,
                        keys = self.metadata.index,
                        images = self.images[Step.REGISTER].values(),
                        masks = self.masks[Step.GLOBAL_GMM].values(),
                        mean_sets = self.mean_sets[Step.GLOBAL_GMM].values(),
                        nok = num_kernel)
        self.updateImages(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot()
    def on_btnScore_clicked(self):
        if Step.LOCAL_GMM not in self.images.keys():
            self.on_btnLocalGmm_clicked()
        print('Scoring..')
        step = Step.SCORE
        self.undoSince(step)
        self.scores[Step.GLOBAL_GMM], self.scores[Step.LOCAL_GMM], self.scores[step] = \
            batch_apply(step,
                        keys = self.metadata.index,
                        global_masks = self.masks[Step.GLOBAL_GMM],
                        local_masks = self.masks[Step.LOCAL_GMM],
                        mean_sets = self.mean_sets[Step.GLOBAL_GMM])
        print('Done.')

    @pyqtSlot()
    def on_btnExport_clicked(self):
        dir = QFileDialog.getExistingDirectory(self, '选择一个目录', '', QFileDialog.ShowDirsOnly)
        new_folder = set(['images', 'masks', 'means','scores']).difference(os.listdir(dir))
        for i in new_folder:
            os.mkdir(os.path.join(dir, i))
        # metadata
        self.metadata.to_csv(os.path.join(dir, 'metadata.csv'))
        # images
        for step, images in self.images.items():
            if step == Step.RAW:
                continue
            sub_folder = os.path.join(dir, 'images', step.name.lower())
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)
            for id, img in images.items():
                cv2.imwrite(os.path.join(sub_folder, id + '.bmp'), img)

        # masks
        for step, masks in self.masks.items():
            sub_folder = os.path.join(dir, 'masks', step.name.lower())
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)
            for id, mask in masks.items():
                cv2.imwrite(os.path.join(sub_folder, id + '.bmp'), mask)
        # mean_sets
        # scores
        for step, table in self.scores.items():
            table.to_csv(os.path.join(dir, 'scores', step.name.lower() + '.csv'))
        pass

    @pyqtSlot(QModelIndex)
    def on_viewSelector_updatePreviewWidget(self, index):
        self.updateImages(index)

