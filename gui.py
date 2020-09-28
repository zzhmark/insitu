import os

import cv2
import pandas as pd
from PyQt5.QtWidgets import QDialog, QFileDialog, QGraphicsScene, QTableView
from PyQt5.QtCore import pyqtSlot, QModelIndex, Qt
from PyQt5.uic import loadUi
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from scipy.sparse import save_npz

from algorithm import batch_apply
from utils import cvimg2qpixmap, fitView, Step


class MainDlg(QDialog):

    def __init__(self):
        super(MainDlg, self).__init__()
        loadUi('dialog.ui', self)
        self.scenes = {i: QGraphicsScene() for i in Step}
        self.viewRaw.setScene(self.scenes[Step.RAW])
        self.viewExtract.setScene(self.scenes[Step.EXTRACT])
        self.viewRegister.setScene(self.scenes[Step.REGISTER])
        self.viewGlobalGmm.setScene(self.scenes[Step.GLOBAL_GMM])
        self.viewLocalGmm.setScene(self.scenes[Step.LOCAL_GMM])
        self.selectorModel = QStandardItemModel()
        self.viewSelector.setModel(self.selectorModel)
        self.images = {}
        self.masks = {}
        self.scores = {}
        self.gmm_models = {}
        self.metadata = pd.DataFrame(columns=[0, 1])
        self.enableStepButtons(False)
        self.supportedFormats = ['.bmp']
        self.viewSelector.setColumnWidths([110] * 4)
        self.scoreTable = QTableView()
        self.scoreTable.setMinimumWidth(387)
        self.gridSelector.addWidget(self.scoreTable, 0, 0, Qt.AlignRight)
        self.tableModel = QStandardItemModel()
        self.scoreTable.setModel(self.tableModel)
        self.tableModel.setHorizontalHeaderLabels(['Gene', 'Stage', 'ID', 'Global', 'Local', 'Hybrid'])
        for i in range(6):
            self.scoreTable.setColumnWidth(i, 55)
        self.scoreTable.setColumnWidth(0, 60)
        self.scoreTable.setColumnWidth(1, 60)
        self.scoreTable.setColumnWidth(2, 100)
        self.scoreTable.setSortingEnabled(True)

    def enableStepButtons(self, enable):
        self.btnExtract.setEnabled(enable)
        self.btnRegister.setEnabled(enable)
        self.btnGlobalGmm.setEnabled(enable)
        self.btnLocalGmm.setEnabled(enable)
        self.btnScore.setEnabled(enable)
        self.btnExport.setEnabled(enable)

    def reset(self):
        self.images.clear()
        self.masks.clear()
        self.scores.clear()
        self.gmm_models.clear()
        self.metadata = pd.DataFrame(columns=[0, 1])
        self.selectorModel.clear()
        for scene in self.scenes.values():
            scene.clear()
        self.enableStepButtons(False)
        self.clearScoreTable()

    def undoSince(self, step):
        for data in [self.images, self.masks, self.gmm_models, self.scores]:
            keys = [key for key in data.keys() if key.value >= step.value]
            for i in keys:
                data.pop(i)
        if step.value <= Step.SCORE.value:
            self.clearScoreTable()

    def updateImages(self, index):
        if index == QModelIndex():
            return
        image_id = self.selectorModel.itemFromIndex(index).text()
        if image_id not in self.metadata.index:
            return
        for scene in self.scenes.values():
            scene.clear()
        for step in self.images.keys():
            image_resize = fitView(self.images[step][image_id], self.scenes[step].views()[0])
            self.scenes[step].addPixmap(cvimg2qpixmap(image_resize))

    @pyqtSlot()
    def on_btnImport_clicked(self):
        print('Importing images..')
        step = Step.RAW
        folder = QFileDialog.getExistingDirectory(self, '选择一个目录', '', QFileDialog.ShowDirsOnly)
        if folder == '':
            return

        self.reset()
        self.images[step] = {}
        for gene in os.listdir(folder):
            gene_dir = os.path.join(folder, gene)
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
                    image_id, ext = os.path.splitext(file)
                    if ext not in self.supportedFormats:
                        continue
                    filename = os.path.join(folder, gene, stage, file)
                    self.images[step][image_id] = cv2.imread(filename, cv2.IMREAD_COLOR)
                    stage_item.appendRow(QStandardItem(image_id))
                    self.metadata = self.metadata.append([[gene, stage, image_id]])
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
                        images=self.images[Step.RAW],
                        kernel=sd_kernel,
                        thr=sd_thr)
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
                        keys=self.metadata.index,
                        images=self.images[Step.EXTRACT].values(),
                        masks=self.masks[Step.EXTRACT].values(),
                        size=size)
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
        self.images[step], self.masks[step], self.gmm_models[step] = \
            batch_apply(step,
                        keys=self.metadata.index,
                        images=self.images[Step.REGISTER].values(),
                        masks=self.masks[Step.REGISTER].values(),
                        nok=num_kernel,
                        patch=patch)
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
        self.images[step], self.masks[step], self.gmm_models[step] = \
            batch_apply(step,
                        keys=self.metadata.index,
                        images=self.images[Step.REGISTER].values(),
                        masks=self.masks[Step.GLOBAL_GMM].values(),
                        global_models=self.gmm_models[Step.GLOBAL_GMM].values(),
                        nok=num_kernel)
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
                        keys=self.metadata.index,
                        global_masks=self.masks[Step.GLOBAL_GMM],
                        local_masks=self.masks[Step.LOCAL_GMM],
                        global_models=self.gmm_models[Step.GLOBAL_GMM])
        self.updateScoreTable(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot()
    def on_btnExport_clicked(self):
        if Step.SCORE not in self.scores.keys():
            self.on_btnScore_clicked()
        folder = QFileDialog.getExistingDirectory(self, '选择一个目录', '', QFileDialog.ShowDirsOnly)
        if folder == "":
            return
        dlg = ExportDialog()
        if dlg.exec_() != QDialog.Accepted:
            return
        new_folder = {'images', 'masks', 'gmm_models', 'scores'}.difference(os.listdir(folder))
        for i in new_folder:
            os.mkdir(os.path.join(folder, i))
        # metadata
        self.metadata.to_csv(os.path.join(folder, 'metadata.csv'))
        # images
        for step, images in self.images.items():
            if step == Step.RAW:
                continue
            sub_folder = os.path.join(folder, 'images', step.name.lower())
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)
            for image_id, image in images.items():
                cv2.imwrite(os.path.join(sub_folder, image_id + '.bmp'), image)

        # masks
        for step, masks in self.masks.items():
            sub_folder = os.path.join(folder, 'masks', step.name.lower())
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)
            for image_id, mask in masks.items():
                cv2.imwrite(os.path.join(sub_folder, image_id + '.bmp'), mask)

        # gmm_models
        for step, gmm_models in self.gmm_models.items():
            sub_folder = os.path.join(folder, 'gmm_models', step.name.lower())
            if not os.path.exists(sub_folder):
                os.mkdir(sub_folder)
            for image_id, gmm_model in gmm_models.items():
                gmm_model.to_csv(os.path.join(sub_folder, image_id + '.csv'))


        # scores
        if dlg.chSparse.isChecked():
            thr_list = [float(dlg.edtGlobalThr.text()),
                        float(dlg.edtLocalThr.text()),
                        float(dlg.edtHybridThr.text())]
            for (step, table), thr in zip(self.scores.items(), thr_list):
                table_thr = table.copy()
                table_thr[table_thr < thr] = 0
                table_sparse = table_thr.astype(pd.SparseDtype('float', 0))
                save_npz(os.path.join(folder, 'scores', step.name.lower() + '.npz'),
                         table_sparse.sparse.to_coo())
            with open(os.path.join(folder, 'scores', 'thresholds.txt'), 'w') as f:
                f.write('GLOBAL=' + dlg.edtGlobalThr.text() + '\n')
                f.write('LOCAL=' + dlg.edtLocalThr.text() + '\n')
                f.write('HYBRID=' + dlg.edtHybridThr.text() + '\n')
        else:
            for step, table in self.scores.items():
                table.to_csv(os.path.join(folder, 'scores', step.name.lower() + '.csv'))

    @pyqtSlot(QModelIndex)
    def on_viewSelector_updatePreviewWidget(self, index):
        self.updateImages(index)
        self.updateScoreTable(index)

    def clearScoreTable(self):
        self.tableModel.setRowCount(0)

    def updateScoreTable(self, index):
        if Step.SCORE not in self.scores.keys():
            return
        if index == QModelIndex():
            return
        ref_id = self.selectorModel.itemFromIndex(index).text()
        if ref_id not in self.metadata.index:
            return
        self.clearScoreTable()
        for ind, row in self.metadata.iterrows():
            self.tableModel.appendRow([QStandardItem(row['gene']),
                                       QStandardItem(row['stage']),
                                       QStandardItem(ind),
                                       QStandardItem(str(self.scores[Step.GLOBAL_GMM][ind][ref_id])),
                                       QStandardItem(str(self.scores[Step.LOCAL_GMM][ind][ref_id])),
                                       QStandardItem(str(self.scores[Step.SCORE][ind][ref_id]))])
        self.scoreTable.setSortingEnabled(True)


class ExportDialog(QDialog):

    def __init__(self):
        super(ExportDialog, self).__init__()
        loadUi('export.ui', self)

    pyqtSlot(bool)
    def on_chSparse_toggled(self, checked):
        self.groupScoreThr.setEnabled(checked)