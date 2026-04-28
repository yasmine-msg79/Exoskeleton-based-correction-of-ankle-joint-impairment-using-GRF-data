"""
main.py — project entry point.
Run this file to launch the GRF Analysis UI.

    python src/main.py
"""

import sys
import os

# Make sure the ui/ folder is on the path
_UI  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))
_SRC = os.path.abspath(os.path.dirname(__file__))

for _p in (_UI, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt5.QtWidgets import QApplication
from main_window import GaitAnalysisUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = GaitAnalysisUI()
    window.show()
    sys.exit(app.exec_())