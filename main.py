from PyQt5.QtWidgets import QApplication
from gui import MainDlg
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlg = MainDlg()
    dlg.show()
    sys.exit(app.exec_())
