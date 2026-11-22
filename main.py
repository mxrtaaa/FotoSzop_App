import sys

from PySide6.QtWidgets import QApplication

from frontend.main_window import ObjectRemoverWindow


def main():
    app = QApplication(sys.argv)
    window = ObjectRemoverWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
