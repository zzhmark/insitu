import sys

from PyQt5.QtWidgets import QApplication

from gui import MainDlg


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlg = MainDlg()
    dlg.show()
    sys.exit(app.exec_())
