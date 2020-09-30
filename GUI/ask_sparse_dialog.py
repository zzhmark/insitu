from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi


class AskSparseDialog(QDialog):

    def __init__(self):
        super(AskSparseDialog, self).__init__()
        loadUi('GUI/rc/ExportDialog.ui', self)

    @pyqtSlot(bool)
    def on_chSparse_toggled(self, checked: bool):
        self.groupScoreThr.setEnabled(checked)
