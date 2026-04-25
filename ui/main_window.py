import sys
import os
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QSpinBox,
    QComboBox,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Add src to sys.path to import backend modules
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from SignalReader import SignalReader
from FeatureExtractor import FeatureExtractor
from Interpreter import Interpreter
from ReferenceBuilder import ReferenceBuilder
from Classifier import Classifier
from training import default_model_path


def _default_data_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))


def _default_model_file():
    return str(default_model_path())


def _interpreter_label_for_class(class_label):
    """Map dataset CLASS_LABEL to the string expected by Interpreter.interpret."""
    s = str(class_label).strip()
    if s == "HC":
        return "normal"
    if s.startswith("A"):
        return "ankle_impairment"
    return s.lower().replace(" ", "_")


class GaitAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GRF Analysis — Ankle / Reference Comparison")
        self.setGeometry(100, 100, 1000, 800)

        self.reader = None
        self.reference = None
        self.classifier = None  # loaded from models/grf_pca_svm_classifier.pkl after training.py
        self.extractor = FeatureExtractor()
        self.interpreter = Interpreter()
        self.model_path = _default_model_file()

        self.init_ui()
        self._load_classifier_model()
        self._load_backend()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel_widget = QWidget()
        left_panel_widget.setFixedWidth(360)
        left_panel = QVBoxLayout(left_panel_widget)
        main_layout.addWidget(left_panel_widget)

        control_group = QGroupBox("Dataset & sample")
        control_layout = QVBoxLayout()

        data_dir = _default_data_dir()
        self.grf_path = os.path.join(data_dir, "GRF_F_V_PRO_left.csv")
        self.meta_path = os.path.join(data_dir, "GRF_metadata.csv")
        control_layout.addWidget(
            QLabel(
                "GRF:\nData/"
                + os.path.basename(self.grf_path)
                + "\n\nMetadata:\nData/"
                + os.path.basename(self.meta_path)
            )
        )
        self.lbl_model_file = QLabel("")
        self.lbl_model_file.setWordWrap(True)
        control_layout.addWidget(self.lbl_model_file)
        # control_layout.addWidget(QLabel("(Run `python src/training.py` to build the classifier.)"))

        self.sample_spin = QSpinBox()
        self.sample_spin.setMinimum(0)
        self.sample_spin.setMaximum(0)
        self.sample_spin.valueChanged.connect(self._on_sample_index_changed)

        form = QFormLayout()
        form.addRow("Sample index:", self.sample_spin)
        control_layout.addLayout(form)

        self.lbl_sample_meta = QLabel("—")
        self.lbl_sample_meta.setWordWrap(True)
        control_layout.addWidget(self.lbl_sample_meta)

        self.combo_interpret = QComboBox()
        self.combo_interpret.addItem("From metadata (auto)", userData="auto")
        self.combo_interpret.addItem("Force: normal", userData="normal")
        self.combo_interpret.addItem("Force: ankle_impairment", userData="ankle_impairment")
        control_layout.addWidget(QLabel("Interpretation label:"))
        control_layout.addWidget(self.combo_interpret)

        self.btn_analyze = QPushButton("Analyze sample")
        self.btn_analyze.clicked.connect(self.run_analysis)
        control_layout.addWidget(self.btn_analyze)

        control_group.setLayout(control_layout)
        left_panel.addWidget(control_group)

        feat_group = QGroupBox("Extracted features")
        feat_layout = QFormLayout()
        self.lbl_first_peak = QLabel("—")
        self.lbl_second_peak = QLabel("—")
        self.lbl_impulse = QLabel("—")
        feat_layout.addRow("First peak:", self.lbl_first_peak)
        feat_layout.addRow("Second peak:", self.lbl_second_peak)
        feat_layout.addRow("Impulse:", self.lbl_impulse)
        feat_group.setLayout(feat_layout)
        left_panel.addWidget(feat_group)

        result_group = QGroupBox("Interpretation")
        result_layout = QVBoxLayout()
        self.lbl_interpretation = QLabel("N/A")
        self.lbl_interpretation.setWordWrap(True)
        self.lbl_interpretation.setStyleSheet("font-weight: bold;")
        result_layout.addWidget(self.lbl_interpretation)
        result_group.setLayout(result_layout)
        left_panel.addWidget(result_group)

        left_panel.addStretch()

        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel)

        self.fig_signals, self.ax_signals = plt.subplots(figsize=(6, 4))
        self.canvas_signals = FigureCanvas(self.fig_signals)
        right_panel.addWidget(self.canvas_signals)

        self.fig_features, self.ax_features = plt.subplots(figsize=(6, 4))
        self.canvas_features = FigureCanvas(self.fig_features)
        right_panel.addWidget(self.canvas_features)

    def _load_backend(self):
        try:
            if not os.path.isfile(self.grf_path) or not os.path.isfile(self.meta_path):
                QMessageBox.warning(
                    self,
                    "Missing data",
                    f"Expected CSV files not found.\n\n{self.grf_path}\n{self.meta_path}",
                )
                return

            self.reader = SignalReader(self.grf_path, self.meta_path)
            ref_builder = ReferenceBuilder()
            self.reference = ref_builder.build(self.reader)

            n = min(len(self.reader.grf_df), len(self.reader.meta_df))
            self.sample_spin.setMaximum(max(0, n - 1))

            self.run_analysis()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load error",
                f"Could not load data or build reference:\n{e}",
            )

    def _load_classifier_model(self):
        self.classifier = None
        if os.path.isfile(self.model_path):
            try:
                self.classifier = Classifier()
                self.classifier.load(self.model_path)
                self.lbl_model_file.setText(
                    "Trained model: loaded\n" + os.path.basename(self.model_path)
                )
            except Exception as e:
                self.lbl_model_file.setText(f"Model load failed: {e}")
        else:
            self.lbl_model_file.setText(
                "Trained model: not found\n(models/grf_pca_svm_classifier.pkl)"
            )

    def _on_sample_index_changed(self, _value):
        if self.reader is None:
            return
        idx = self.sample_spin.value()
        n = min(len(self.reader.grf_df), len(self.reader.meta_df))
        if idx < 0 or idx >= n:
            return
        label = self.reader.meta_df.iloc[idx]["CLASS_LABEL"]
        self.lbl_sample_meta.setText(f"CLASS_LABEL: {label}")

    def _interpret_label_for_index(self, index):
        mode = self.combo_interpret.currentData()
        if mode != "auto":
            return mode
        class_label = self.reader.meta_df.iloc[index]["CLASS_LABEL"]
        return _interpreter_label_for_class(class_label)

    def run_analysis(self):
        if self.reader is None or self.reference is None:
            return
        try:
            idx = self.sample_spin.value()
            sample = self.reader.get_sample(idx)
            features = self.extractor.extract(sample)
            interp_label = self._interpret_label_for_index(idx)
            result = self.interpreter.interpret(features, interp_label)

            self.lbl_sample_meta.setText(
                f"Index: {idx}\nCLASS_LABEL: {sample['label']}\n"
                f"Affected side: {sample['Affected_Limb']}"
            )

            self.lbl_first_peak.setText(f"{features['first_peak']:.4f}")
            self.lbl_second_peak.setText(f"{features['second_peak']:.4f}")
            self.lbl_impulse.setText(f"{features['impulse']:.4f}")

            rule_text = str(result)
            if self.classifier is not None and getattr(self.classifier, "fitted", False):
                ml_label, margin = self.classifier.predict_row(sample["grf"])
                interp_text = (
                    f"PCA+SVM: {ml_label} (score={margin:.3f})\n"
                    f"Rule-based: {rule_text}"
                )
            else:
                interp_text = (
                    "PCA+SVM: (no model — run python src/training.py)\n"
                    f"Rule-based: {rule_text}"
                )
            self.lbl_interpretation.setText(interp_text)

            self._update_plots(sample, features)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Analysis failed:\n{e}",
            )

    def _update_plots(self, sample, features):
        t = sample["time"]
        grf = sample["grf"]

        self.ax_signals.clear()
        self.ax_signals.plot(t, grf, label="Selected GRF")
        self.ax_signals.plot(
            t,
            self.reference,
            linestyle="--",
            label="Reference (healthy mean)",
        )
        self.ax_signals.set_title("GRF vs reference")
        self.ax_signals.set_xlabel("Gait cycle (normalized)")
        self.ax_signals.set_ylabel("Force (BW-normalized)")
        self.ax_signals.legend()
        self.ax_signals.grid(True)
        self.fig_signals.tight_layout()
        self.canvas_signals.draw()

        self.ax_features.clear()
        names = ["First peak", "Second peak", "Impulse"]
        vals = [features["first_peak"], features["second_peak"], features["impulse"]]
        self.ax_features.bar(names, vals, color=["#4C72B0", "#55A868", "#C44E52"])
        self.ax_features.set_title("Extracted features")
        self.ax_features.grid(True, axis="y")
        self.fig_features.tight_layout()
        self.canvas_features.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = GaitAnalysisUI()
    window.show()
    sys.exit(app.exec_())
