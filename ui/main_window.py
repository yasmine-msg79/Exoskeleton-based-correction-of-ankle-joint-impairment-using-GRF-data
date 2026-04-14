import sys
import os
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QRadioButton, QDoubleSpinBox, QLabel, 
                             QComboBox, QPushButton, QGroupBox, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Add src to sys.path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from main import main as GaitPipeline

class GaitAnalysisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gait Analysis - Hip Exo Control")
        # Ensure minimum size to prevent squishing
        self.setGeometry(100, 100, 1000, 800)
        
        self.pipeline = GaitPipeline()
        
        self.init_ui()
        
    def init_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left Panel (Controls and Status)
        left_panel_widget = QWidget()
        left_panel_widget.setFixedWidth(350)
        left_panel = QVBoxLayout(left_panel_widget)
        main_layout.addWidget(left_panel_widget)
        
        # --- Control Group ---
        control_group = QGroupBox("Configuration")
        control_layout = QVBoxLayout()
        
        # Radio Buttons
        self.radio_manual = QRadioButton("Manual Input")
        self.radio_scenario = QRadioButton("Scenarios")
        self.radio_manual.setChecked(True) # Default
        
        self.radio_manual.toggled.connect(self.toggle_input_mode)
        
        control_layout.addWidget(self.radio_manual)
        control_layout.addWidget(self.radio_scenario)
        
        # Manual Inputs
        self.manual_widget = QWidget()
        manual_form = QFormLayout(self.manual_widget)
        
        self.heel_input = QDoubleSpinBox()
        self.heel_input.setRange(0.0, 5.0)
        self.heel_input.setSingleStep(0.1)
        self.heel_input.setValue(1.0)
        
        self.toe_input = QDoubleSpinBox()
        self.toe_input.setRange(0.0, 5.0)
        self.toe_input.setSingleStep(0.1)
        self.toe_input.setValue(0.35)
        
        self.hip_input = QDoubleSpinBox()
        self.hip_input.setRange(0.0, 90.0)
        self.hip_input.setSingleStep(1.0)
        self.hip_input.setValue(15.0)
        
        manual_form.addRow("Heel Peak:", self.heel_input)
        manual_form.addRow("Toe Peak:", self.toe_input)
        manual_form.addRow("Hip Range:", self.hip_input)
        
        control_layout.addWidget(self.manual_widget)
        
        # Scenarios Dropdown
        self.scenario_widget = QWidget()
        scenario_layout = QVBoxLayout(self.scenario_widget)
        scenario_layout.setContentsMargins(0, 0, 0, 0)
        self.scenario_dropdown = QComboBox()
        self.scenario_dropdown.addItems(["Normal", "Knee Impairment", "Hip Impairment"])
        scenario_layout.addWidget(QLabel("Select Scenario:"))
        scenario_layout.addWidget(self.scenario_dropdown)
        
        control_layout.addWidget(self.scenario_widget)
        self.scenario_widget.setVisible(False) # Hide initially
        
        # Analyze Button
        self.btn_analyze = QPushButton("Analyze Gait")
        self.btn_analyze.clicked.connect(self.run_analysis)
        control_layout.addWidget(self.btn_analyze)
        
        control_group.setLayout(control_layout)
        left_panel.addWidget(control_group)
        
        # --- Phase Status Group ---
        phase_group = QGroupBox("Gait Phases Status")
        self.phase_layout = QFormLayout()
        
        self.phase_labels = {
            "heel_strike_done": QLabel("⏳"),
            "toe_off_done": QLabel("⏳"),
            "mid_stance_phase_done": QLabel("⏳"),
            "swing_phase_done": QLabel("⏳")
        }
        
        self.phase_layout.addRow("Heel Strike:", self.phase_labels["heel_strike_done"])
        self.phase_layout.addRow("Toe Off:", self.phase_labels["toe_off_done"])
        self.phase_layout.addRow("Mid Stance:", self.phase_labels["mid_stance_phase_done"])
        self.phase_layout.addRow("Swing Phase:", self.phase_labels["swing_phase_done"])
        
        phase_group.setLayout(self.phase_layout)
        left_panel.addWidget(phase_group)
        
        # --- Classification and Correction Result Add-on ---
        result_group = QGroupBox("Analysis Results")
        result_layout = QVBoxLayout()
        self.lbl_classification = QLabel("Classification: N/A")
        self.lbl_classification.setWordWrap(True)
        self.lbl_classification.setStyleSheet("font-weight: bold;")
        
        self.lbl_correction = QLabel("Correction: N/A")
        self.lbl_correction.setWordWrap(True)
        
        result_layout.addWidget(self.lbl_classification)
        result_layout.addWidget(self.lbl_correction)
        result_group.setLayout(result_layout)
        left_panel.addWidget(result_group)

        left_panel.addStretch()
        
        # Right Panel (Plots)
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel)
        
        # Matplotlib Figures
        self.fig_signals, self.ax_signals = plt.subplots(figsize=(6, 4))
        self.canvas_signals = FigureCanvas(self.fig_signals)
        right_panel.addWidget(self.canvas_signals)
        
        self.fig_correction, self.ax_correction = plt.subplots(figsize=(6, 4))
        self.canvas_correction = FigureCanvas(self.fig_correction)
        right_panel.addWidget(self.canvas_correction)
        
        # Run Initial Analysis so graphs aren't empty
        self.run_analysis()
        
    def toggle_input_mode(self):
        is_manual = self.radio_manual.isChecked()
        self.manual_widget.setVisible(is_manual)
        self.scenario_widget.setVisible(not is_manual)

    def run_analysis(self):
        try:
            if self.radio_manual.isChecked():
                heel_peak = self.heel_input.value()
                toe_peak = self.toe_input.value()
                hip_range = self.hip_input.value()
            else:
                scenario = self.scenario_dropdown.currentText()
                if scenario == "Normal":
                    heel_peak = 1.0
                    toe_peak = 0.5
                    hip_range = 30.0
                elif scenario == "Knee Impairment":
                    heel_peak = 0.3
                    toe_peak = 0.3
                    hip_range = 30.0
                elif scenario == "Hip Impairment":
                    heel_peak = 1.0
                    toe_peak = 0.5
                    hip_range = 10.0
            
            # Run backend
            results = self.pipeline.run(heel_peak, toe_peak, hip_range)
            
            # Update Phase Status
            states = results["phase_states"]
            for key, label in self.phase_labels.items():
                if states.get(key, False):
                    label.setText("✅  Done")
                    label.setStyleSheet("color: green; font-weight: bold;")
                else:
                    label.setText("❌  Missed")
                    label.setStyleSheet("color: red; font-weight: bold;")
                    
            # Update text results
            classes = ", ".join(results["classification"])
            corrs = ", ".join(results["correction"])
            self.lbl_classification.setText(f"Classification: {classes}")
            self.lbl_correction.setText(f"Correction: {corrs}")
            
            # Update Plots
            self.update_plots(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during analysis:\n{str(e)}")

    def update_plots(self, results):
        signals = results["signals"]
        time = signals["time"]
        heel = signals["heel"]
        toe = signals["toe"]
        hip = signals["hip_measured"]
        corrected_hip = results["corrected_hip"]
        
        # Plot 1: 3 signals (Heel, Toe, Hip)
        self.ax_signals.clear()
        self.ax_signals.plot(time, heel, label="Heel")
        self.ax_signals.plot(time, toe, label="Toe")
        self.ax_signals.plot(time, hip, label="Hip (Measured)")
        self.ax_signals.set_title("Simulated Gait Signals")
        self.ax_signals.set_xlabel("Time (s)")
        self.ax_signals.set_ylabel("Amplitude")
        self.ax_signals.legend()
        self.ax_signals.grid(True)
        self.fig_signals.tight_layout()
        self.canvas_signals.draw()
        
        # Plot 2: Original Hip and Corrected Hip
        self.ax_correction.clear()
        self.ax_correction.plot(time, hip, label="Original Hip")
        self.ax_correction.plot(time, corrected_hip, label="Corrected Hip", linestyle='--')
        self.ax_correction.set_title("Hip Correction")
        self.ax_correction.set_xlabel("Time (s)")
        self.ax_correction.set_ylabel("Amplitude")
        self.ax_correction.legend()
        self.ax_correction.grid(True)
        self.fig_correction.tight_layout()
        self.canvas_correction.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = GaitAnalysisUI()
    window.show()
    sys.exit(app.exec_())
