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
from .ask_sparse_dialog import AskSparseDialog


class MainDialog(QDialog):

    def __init__(self):
        super(MainDialog, self).__init__()
        loadUi('GUI/rc/MainDialog.ui', self)
        self.enableStepButtons(False)

        # Model/View setup.
        # Graphics.
        self.scenes = {i: QGraphicsScene() for i in Step}
        self.viewRaw.setScene(self.scenes[Step.RAW])
        self.viewExtract.setScene(self.scenes[Step.EXTRACT])
        self.viewRegister.setScene(self.scenes[Step.REGISTER])
        self.viewGlobalGmm.setScene(self.scenes[Step.GLOBAL_GMM])
        self.viewLocalGmm.setScene(self.scenes[Step.LOCAL_GMM])
        # Columns and tables.
        self.selectorModel = QStandardItemModel()
        self.viewSelector.setModel(self.selectorModel)
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

        # Data init.
        self.images = {}
        self.masks = {}
        self.labels = {}
        self.scores = {}
        self.gmm_models = {}
        self.metadata = pd.DataFrame(columns=[0, 1])

        # Signal/Slot Connection
        self.scoreTable.clicked.connect(self.on_scoreTable_clicked)

    def enableStepButtons(self, enable: bool):
        self.btnExtract.setEnabled(enable)
        self.btnRegister.setEnabled(enable)
        self.btnGlobalGmm.setEnabled(enable)
        self.btnLocalGmm.setEnabled(enable)
        self.btnHybrid.setEnabled(enable)
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
        self.tableModel.setRowCount(0)

    def undoSince(self, step: Step):
        for data in [self.images, self.masks, self.labels, self.gmm_models, self.scores]:
            keys = [key for key in data.keys() if key.value >= step.value]
            for i in keys:
                data.pop(i)

    def updateImages(self, index: str):
        for scene in self.scenes.values():
            scene.clear()
        for step in self.images.keys():
            image_resize = fitView(self.images[step][index], self.scenes[step].views()[0])
            self.scenes[step].addPixmap(cvimg2qpixmap(image_resize))

    @pyqtSlot()
    def on_btnImport_clicked(self):
        print('Importing images..')
        step = Step.RAW
        folder = QFileDialog.getExistingDirectory(self, '选择一个目录', '', QFileDialog.ShowDirsOnly)
        if folder == '':
            print('Aborted.')
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
                    if ext not in ['.bmp']:
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
        try:
            sd_kernel = (int(sd_kernel_size_input),) * 2
            sd_thr = float(sd_thr_input)
        except ValueError:
            print('Invalid input!')
            return
        print('Extracting..')
        step = Step.EXTRACT
        self.undoSince(step)
        self.images[step], self.masks[step] = batch_apply(step,
                                                          keys=self.metadata.index,
                                                          images=self.images[Step.RAW].values(),
                                                          kernel=sd_kernel,
                                                          thr=sd_thr)
        self.on_viewSelector_updatePreviewWidget(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot(bool)
    def on_btnRegister_clicked(self):
        width_input = self.edtWidth.text()
        height_input = self.edtHeight.text()
        try:
            size = (int(width_input), int(height_input))
        except ValueError:
            print('Invalid input!')
            return
        if Step.EXTRACT not in self.images.keys():
            self.on_btnExtract_clicked()
        print('Registering..')
        step = Step.REGISTER
        self.undoSince(step)
        self.images[step], self.masks[step] = batch_apply(step,
                                                          keys=self.metadata.index,
                                                          images=self.images[Step.EXTRACT].values(),
                                                          masks=self.masks[Step.EXTRACT].values(),
                                                          size=size)
        self.on_viewSelector_updatePreviewWidget(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot()
    def on_btnGlobalGmm_clicked(self):
        num_kernel_input = self.edtGlobalNumKernel.text()
        pooling_rate_input = self.edtPoolingRate.text()
        try:
            num_kernel = int(num_kernel_input)
            patch = (int(pooling_rate_input),) * 2
        except ValueError:
            print('Invalid input!')
            return
        if Step.REGISTER not in self.images.keys():
            self.on_btnRegister_clicked()
        print('Performing Global GMM..')
        step = Step.GLOBAL_GMM
        self.undoSince(step)
        self.images[step], self.masks[step], self.labels[step], self.gmm_models[step], self.scores[step] = \
            batch_apply(step,
                        keys=self.metadata.index,
                        images=self.images[Step.REGISTER].values(),
                        masks=self.masks[Step.REGISTER].values(),
                        nok=num_kernel,
                        patch=patch)
        self.on_viewSelector_updatePreviewWidget(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot()
    def on_btnLocalGmm_clicked(self):
        num_kernel_input = self.edtGlobalNumKernel.text()
        global_cutoff_input = self.edtGlobalCutoff.text()
        try:
            num_kernel = int(num_kernel_input)
            global_cutoff = float(global_cutoff_input)
        except ValueError:
            print('Invalid input!')
            return
        if Step.GLOBAL_GMM not in self.images.keys():
            self.on_btnGlobalGmm_clicked()
        print('Performing Local GMM..')
        step = Step.LOCAL_GMM
        self.undoSince(step)
        self.images[step], self.labels[step], self.gmm_models[step], self.scores[step] = \
            batch_apply(step,
                        keys=self.metadata.index,
                        images=self.images[Step.REGISTER].values(),
                        labels=self.labels[Step.GLOBAL_GMM].values(),
                        models=self.gmm_models[Step.GLOBAL_GMM].values(),
                        scores=self.scores[Step.GLOBAL_GMM],
                        cutoff=global_cutoff,
                        nok=num_kernel)
        self.on_viewSelector_updatePreviewWidget(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot()
    def on_btnHybrid_clicked(self):
        if Step.LOCAL_GMM not in self.images.keys():
            self.on_btnLocalGmm_clicked()
        print('Scoring..')
        step = Step.HYBRID
        self.undoSince(step)
        self.scores[step] = batch_apply(step,
                                        global_scores=self.scores[Step.GLOBAL_GMM],
                                        local_scores=self.scores[Step.LOCAL_GMM])
        self.on_viewSelector_updatePreviewWidget(self.viewSelector.currentIndex())
        print('Done.')

    @pyqtSlot()
    def on_btnExport_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, '选择一个目录', '', QFileDialog.ShowDirsOnly)
        if folder == "":
            print('Aborted.')
            return
        dlg = AskSparseDialog()
        if dlg.exec_() != QDialog.Accepted:
            print('Aborted.')
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
                f.write('GLOBAL_CUTOFF=' + self.edtGlobalCutoff.text() + '\n')
                f.write('LOCAL=' + dlg.edtLocalThr.text() + '\n')
                f.write('HYBRID=' + dlg.edtHybridThr.text() + '\n')
        else:
            for step, table in self.scores.items():
                table.to_csv(os.path.join(folder, 'scores', step.name.lower() + '.csv'))

    @pyqtSlot(QModelIndex)
    def on_viewSelector_updatePreviewWidget(self, index: QModelIndex):
        if index != QModelIndex():
            image_id = self.selectorModel.itemFromIndex(index).text()
            if image_id in self.metadata.index:
                self.updateImages(image_id)
                self.updateScoreTable(image_id)

    @pyqtSlot(QModelIndex)
    def on_scoreTable_clicked(self, index: QModelIndex):
        if index != QModelIndex():
            image_id = self.tableModel.itemFromIndex(index).text()
            if image_id in self.metadata.index:
                self.updateImages(image_id)

    def updateScoreTable(self, index: str):
        self.tableModel.setRowCount(0)
        for ind, row in self.metadata.iterrows():
            new_row = [QStandardItem(row['gene']),
                       QStandardItem(row['stage']),
                       QStandardItem(ind)]
            for scores in self.scores.values():
                score = max([table[ind][index] for table in scores])
                new_row.append(QStandardItem(str(score)))
            self.tableModel.appendRow(new_row)
        self.scoreTable.setSortingEnabled(True)
