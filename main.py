import sys

from PyQt5.QtWidgets import QApplication

from GUI.main_dialog import MainDialog


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dlg = MainDialog()
    dlg.show()
    sys.exit(app.exec_())
