# -*- coding: utf-8 -*-
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
    QScrollArea,
    QSizePolicy,
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
from Correction import Correction
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
        self.setWindowTitle("GRF Analysis — Gait Correction & Error Metrics")
        self.setGeometry(80, 60, 1400, 900)

        self.reader = None
        self.reference = None
        self.classifier = None
        self.extractor = FeatureExtractor()
        self.interpreter = Interpreter()
        self.corrector = Correction()
        self.model_path = _default_model_file()


        self.init_ui()
        self._load_classifier_model()
        self._load_backend()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root = QHBoxLayout(central_widget)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Left sidebar (scrollable) ─────────────────────────────────────────
        sidebar_content = QWidget()
        sidebar_content.setFixedWidth(340)
        sidebar_vbox = QVBoxLayout(sidebar_content)
        sidebar_vbox.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar_content)
        scroll.setFixedWidth(358)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll)

        # ── Dataset & sample group ────────────────────────────────────────────
        data_group = QGroupBox("Dataset & sample")
        data_layout = QVBoxLayout()

        data_dir = _default_data_dir()
        self.grf_path  = os.path.join(data_dir, "GRF_F_V_PRO_left.csv")
        self.meta_path = os.path.join(data_dir, "GRF_metadata.csv")
        data_layout.addWidget(
            QLabel(
                "GRF:\nData/" + os.path.basename(self.grf_path) +
                "\n\nMetadata:\nData/" + os.path.basename(self.meta_path)
            )
        )
        self.lbl_model_file = QLabel("")
        self.lbl_model_file.setWordWrap(True)
        data_layout.addWidget(self.lbl_model_file)

        self.sample_spin = QSpinBox()
        self.sample_spin.setMinimum(0)
        self.sample_spin.setMaximum(0)
        self.sample_spin.valueChanged.connect(self._on_sample_index_changed)

        form0 = QFormLayout()
        form0.addRow("Sample index:", self.sample_spin)
        data_layout.addLayout(form0)

        self.lbl_sample_meta = QLabel("—")
        self.lbl_sample_meta.setWordWrap(True)
        data_layout.addWidget(self.lbl_sample_meta)

        self.combo_interpret = QComboBox()
        self.combo_interpret.addItem("From metadata (auto)", userData="auto")
        self.combo_interpret.addItem("Force: normal",            userData="normal")
        self.combo_interpret.addItem("Force: ankle_impairment",  userData="ankle_impairment")
        data_layout.addWidget(QLabel("Interpretation label:"))
        data_layout.addWidget(self.combo_interpret)

        self.btn_analyze = QPushButton("Analyze sample")
        self.btn_analyze.clicked.connect(self.run_analysis)
        data_layout.addWidget(self.btn_analyze)
        data_group.setLayout(data_layout)
        sidebar_vbox.addWidget(data_group)

        # ── Extracted features group ──────────────────────────────────────────
        feat_group = QGroupBox("Extracted features")
        feat_layout = QFormLayout()
        self.lbl_first_peak  = QLabel("—")
        self.lbl_second_peak = QLabel("—")
        self.lbl_impulse     = QLabel("—")
        feat_layout.addRow("First peak:",  self.lbl_first_peak)
        feat_layout.addRow("Second peak:", self.lbl_second_peak)
        feat_layout.addRow("Impulse:",     self.lbl_impulse)
        feat_group.setLayout(feat_layout)
        sidebar_vbox.addWidget(feat_group)

        # ── Interpretation group ──────────────────────────────────────────────
        result_group = QGroupBox("Interpretation")
        result_layout = QVBoxLayout()
        self.lbl_interpretation = QLabel("N/A")
        self.lbl_interpretation.setWordWrap(True)
        self.lbl_interpretation.setStyleSheet("font-weight: bold;")
        result_layout.addWidget(self.lbl_interpretation)
        result_group.setLayout(result_layout)
        sidebar_vbox.addWidget(result_group)

        # ── Error metrics group ───────────────────────────────────────────────
        err_group = QGroupBox("GRF Error Metrics (vs. HC reference)")
        err_layout = QFormLayout()
        err_layout.setSpacing(4)

        self.lbl_rmse        = QLabel("—")
        self.lbl_mae         = QLabel("—")
        self.lbl_delta_f1    = QLabel("—")
        self.lbl_delta_f2    = QLabel("—")
        self.lbl_delta_imp   = QLabel("—")
        self.lbl_si_f1       = QLabel("—")
        self.lbl_si_f2       = QLabel("—")
        self.lbl_severity    = QLabel("—")

        err_layout.addRow("RMSE [BW]:",        self.lbl_rmse)
        err_layout.addRow("MAE [BW]:",         self.lbl_mae)
        err_layout.addRow("ΔF₁ [BW]:",         self.lbl_delta_f1)
        err_layout.addRow("ΔF₂ [BW]:",         self.lbl_delta_f2)
        err_layout.addRow("ΔImpulse [BW·s]:",  self.lbl_delta_imp)
        err_layout.addRow("SI-F₁ [%]:",        self.lbl_si_f1)
        err_layout.addRow("SI-F₂ [%]:",        self.lbl_si_f2)
        err_layout.addRow("Severity (0–1):",   self.lbl_severity)
        err_group.setLayout(err_layout)
        sidebar_vbox.addWidget(err_group)

        # ── Correction gains group ────────────────────────────────────────────
        gain_group = QGroupBox("Applied Correction Gains (Kp × severity)")
        gain_layout = QFormLayout()
        gain_layout.setSpacing(4)
        self.lbl_kp_load    = QLabel("—")
        self.lbl_kp_mid     = QLabel("—")
        self.lbl_kp_push    = QLabel("—")
        gain_layout.addRow("Kp loading (0–20%):",  self.lbl_kp_load)
        gain_layout.addRow("Kp mid-stance (20–60%):", self.lbl_kp_mid)
        gain_layout.addRow("Kp push-off (60–100%):",  self.lbl_kp_push)
        gain_group.setLayout(gain_layout)
        sidebar_vbox.addWidget(gain_group)

        # ── Suggestions group ─────────────────────────────────────────────────
        sugg_group = QGroupBox("Clinical Suggestions")
        sugg_layout = QVBoxLayout()
        self.lbl_suggestions = QLabel("—")
        self.lbl_suggestions.setWordWrap(True)
        self.lbl_suggestions.setStyleSheet("font-size: 12px;")
        sugg_layout.addWidget(self.lbl_suggestions)
        sugg_group.setLayout(sugg_layout)
        sidebar_vbox.addWidget(sugg_group)

        sidebar_vbox.addStretch()

        # ── Right panel — plots ───────────────────────────────────────────────
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)
        root.addLayout(right_panel)

        # Plot 1: GRF vs reference
        self.fig_signals, self.ax_signals = plt.subplots(figsize=(7, 3))
        self.canvas_signals = FigureCanvas(self.fig_signals)
        self.canvas_signals.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.canvas_signals)

        # Plot 2: features bar chart
        self.fig_features, self.ax_features = plt.subplots(figsize=(7, 2.5))
        self.canvas_features = FigureCanvas(self.fig_features)
        self.canvas_features.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.canvas_features)

        # Plot 3: corrected GRF
        self.fig_corrected, self.ax_corrected = plt.subplots(figsize=(7, 3))
        self.canvas_corrected = FigureCanvas(self.fig_corrected)
        self.canvas_corrected.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.canvas_corrected)

    # ─────────────────────────────────────────────────────────────────────────
    # Backend loading
    # ─────────────────────────────────────────────────────────────────────────

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
                self, "Load error",
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

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────────
    # Core analysis
    # ─────────────────────────────────────────────────────────────────────────

    def run_analysis(self):
        if self.reader is None or self.reference is None:
            return
        try:
            idx    = self.sample_spin.value()
            sample = self.reader.get_sample(idx)
            grf_m  = sample["grf"]

            # ── Feature extraction & interpretation ───────────────────────────
            features    = self.extractor.extract(sample)
            interp_label = self._interpret_label_for_index(idx)
            result      = self.interpreter.interpret(features, interp_label)

            self.lbl_sample_meta.setText(
                f"Index: {idx}\nCLASS_LABEL: {sample['label']}\n"
                f"Affected side: {sample['Affected_Limb']}"
            )
            self.lbl_first_peak.setText(f"{features['first_peak']:.4f}")
            self.lbl_second_peak.setText(f"{features['second_peak']:.4f}")
            self.lbl_impulse.setText(f"{features['impulse']:.4f}")

            rule_text = str(result)
            if self.classifier is not None and getattr(self.classifier, "fitted", False):
                ml_label, margin = self.classifier.predict_row(grf_m)
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

            # ── Error & correction ────────────────────────────────────────────
            grf_corrected, info = self.corrector.apply_correction(grf_m, self.reference)
            self._update_error_panel(info)

            # Suggestions use raw CLASS_LABEL so HC maps to "no correction"
            suggestions = self.corrector.suggest(info, sample["label"])
            self.lbl_suggestions.setText(suggestions)

            # ── Plots ─────────────────────────────────────────────────────────
            self._update_plots(sample, features, grf_corrected)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed:\n{e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Panel / plot helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _update_error_panel(self, info: dict):
        """Populate error-metric and gain labels."""

        def _col(val: float, lo: float, hi: float) -> str:
            """Return a colour string based on thresholds."""
            if abs(val) < lo:
                return "#1a7a1a"   # dark green  — OK / within normal range
            if abs(val) < hi:
                return "#b85c00"   # dark orange — moderate deviation
            return "#c0000a"       # dark red    — high deviation

        rmse     = info.get("rmse",         0.0)
        mae      = info.get("mae",          0.0)
        delta_f1 = info.get("delta_f1",     0.0)
        delta_f2 = info.get("delta_f2",     0.0)
        d_imp    = info.get("delta_impulse", 0.0)
        si_f1    = info.get("si_f1",        0.0)
        si_f2    = info.get("si_f2",        0.0)
        severity = info.get("severity",     0.0)

        def _lbl(widget, text, colour):
            widget.setText(text)
            widget.setStyleSheet(f"color: {colour}; font-weight: bold;")

        _lbl(self.lbl_rmse,      f"{rmse:.4f}",      _col(rmse,     0.05, 0.15))
        _lbl(self.lbl_mae,       f"{mae:.4f}",        _col(mae,      0.03, 0.10))
        _lbl(self.lbl_delta_f1,  f"{delta_f1:+.4f}",  _col(delta_f1, 0.05, 0.15))
        _lbl(self.lbl_delta_f2,  f"{delta_f2:+.4f}",  _col(delta_f2, 0.05, 0.15))
        _lbl(self.lbl_delta_imp, f"{d_imp:+.3f}",     _col(d_imp,    0.50, 2.00))
        _lbl(self.lbl_si_f1,     f"{si_f1:.1f} %",    _col(si_f1,    5.0,  15.0))
        _lbl(self.lbl_si_f2,     f"{si_f2:.1f} %",    _col(si_f2,    5.0,  15.0))
        _lbl(self.lbl_severity,  f"{severity:.3f}",   _col(severity, 0.15, 0.50))

        kp_l = info.get("kp_loading_applied",  0.0)
        kp_m = info.get("kp_mid_applied",      0.0)
        kp_p = info.get("kp_pushoff_applied",  0.0)
        self.lbl_kp_load.setText(f"{kp_l:.3f}")
        self.lbl_kp_mid.setText(f"{kp_m:.3f}")
        self.lbl_kp_push.setText(f"{kp_p:.3f}")

    def _update_plots(self, sample: dict, features: dict, grf_corrected: np.ndarray):
        t   = sample["time"]
        grf = sample["grf"]

        # ── Plot 1: GRF vs reference ──────────────────────────────────────────
        ax = self.ax_signals
        ax.clear()
        ax.plot(t, grf,            label="Patient GRF")
        ax.plot(t, self.reference, linestyle="--", label="Reference (HC mean)")
        ax.set_title("GRF vs HC Reference")
        ax.set_xlabel("Gait cycle (normalised)")
        ax.set_ylabel("Force (BW-normalised)")
        ax.legend()
        ax.grid(True)
        self.fig_signals.tight_layout()
        self.canvas_signals.draw()

        # ── Plot 2: Features bar chart ────────────────────────────────────────
        ax2 = self.ax_features
        ax2.clear()
        names = ["First peak", "Second peak", "Impulse"]
        vals  = [features["first_peak"], features["second_peak"], features["impulse"]]
        ax2.bar(names, vals, color=["#4C72B0", "#55A868", "#C44E52"])
        ax2.set_title("Extracted Features")
        ax2.grid(True, axis="y")
        self.fig_features.tight_layout()
        self.canvas_features.draw()

        # ── Plot 3: Corrected GRF ─────────────────────────────────────────────
        ax3 = self.ax_corrected
        ax3.clear()
        ax3.plot(t, grf,            alpha=0.7, label="Patient GRF (measured)")
        ax3.plot(t, self.reference, linestyle="--", alpha=0.7, label="Reference (HC mean)")
        ax3.plot(t, grf_corrected,  linewidth=2.2, label="Corrected GRF (AAN)")
        ax3.fill_between(t, grf, grf_corrected,
                         where=(grf_corrected != grf),
                         alpha=0.15, label="_correction area")
        ax3.set_title("Corrected GRF  (grf_corrected = grf_measured + Kp(t) * e(t))")
        ax3.set_xlabel("Gait cycle (normalised)")
        ax3.set_ylabel("Force (BW-normalised)")
        ax3.legend()
        ax3.grid(True)
        self.fig_corrected.tight_layout()
        self.canvas_corrected.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = GaitAnalysisUI()
    window.show()
    sys.exit(app.exec_())
