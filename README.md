# Exoskeleton-Based Gait Analysis and Correction System

A machine learning-powered gait analysis system that classifies gait patterns, computes biomechanical error metrics, and applies **Assist-As-Needed (AAN)** corrective strategies for lower-limb exoskeleton control using Ground Reaction Force (GRF) data.

## Project Overview

This system provides a complete pipeline for analyzing vertical GRF waveforms, classifying gait as normal vs. impaired using PCA + SVM, computing quantified error metrics, and generating phase-dependent corrective torque profiles. Designed to support hip exoskeleton control by identifying gait deficits and adapting correction to impairment severity.

**Key Features:**

- PCA + SVM classification with balanced class weights
- Healthy reference waveform construction from control subjects
- Quantified error metrics (RMSE, peak deficits, symmetry indices)
- Severity scoring mapped to clinical thresholds
- Phase-dependent Assist-As-Needed correction
- Interactive PyQt5 GUI with real-time visualization


## Project Structure

```plaintext
├── Data/
│   ├── GRF_F_V_PRO_left.csv      # GRF waveforms (101 samples per row)
│   └── GRF_metadata.csv          # Subject metadata
├── models/
│   └── grf_pca_svm_classifier.pkl
├── src/
│   ├── main.py                   # Entry point
│   ├── training.py               # Model training
│   ├── SignalReader.py           # CSV loading
│   ├── FeatureExtractor.py       # Feature extraction
│   ├── ReferenceBuilder.py       # Reference construction
│   ├── Classifier.py             # PCA + SVM classification
│   ├── Interpreter.py            # Rule-based interpretation
│   └── Correction.py             # Error metrics and AAN correction
└── ui/
    └── main_window.py            # PyQt5 GUI
```

## Getting Started

### Prerequisites

Python 3.8+, numpy, pandas, scikit-learn, matplotlib, PyQt5

### Installation

```shellscript
git clone https://github.com/yourusername/Exoskeleton-GRF-Correction.git
cd Exoskeleton-GRF-Correction
pip install -r requirements.txt
```

### Training the Classifier

```shellscript
python src/training.py
```

### Launching the GUI

```shellscript
python src/main.py
```

## Module Documentation

**SignalReader** — Loads GRF waveforms and metadata from CSV files. Returns time-normalized 101-point waveforms with class labels.

**FeatureExtractor** — Extracts biomechanical features: first peak (weight acceptance), second peak (push-off), and impulse (total stance force).

**ReferenceBuilder** — Constructs normative reference by averaging all healthy control (HC) samples.

**Classifier** — Binary classification using StandardScaler + PCA (95% variance) + SVM with RBF kernel. Returns "normal" or "impaired" with confidence margin.

**Interpreter** — Maps classification results to clinical labels based on dataset categories (HC, Ankle, Knee, Hip, Other).

**Correction** — Computes error metrics and applies phase-dependent AAN correction with configurable gains.

## Clinical Context

This system is grounded in established biomechanical literature including Perry & Burnfield (2010) for gait analysis fundamentals, Robinson et al. (1987) for the symmetry index formula, Kirtley (2006) for clinical thresholds, and Banala et al. (2009) for the Assist-As-Needed paradigm.

Typical healthy GRF pattern shows first peak (~1.0-1.2 BW) during weight acceptance, trough (~0.7-0.8 BW) during mid-stance, and second peak (~1.0-1.2 BW) during push-off.

## Error Metrics Explained

| Metric | Description
|-----|-----
| **RMSE** | Overall waveform deviation (BW units)
| **Delta-F1** | First peak deficit (+ = underloading)
| **Delta-F2** | Second peak deficit (+ = weak push-off)
| **SI-F1/SI-F2** | Symmetry indices using Robinson formula: 200 | X1-X2 | /(X1+X2)
| **Severity** | Normalized 0-1 score: min(1, RMSE/0.3)


**Severity Thresholds:** `<0.33 mild, 0.33-0.66 moderate, >`0.66 severe

## Assist-As-Needed Correction

The correction follows: `GRF_corrected(t) = GRF_measured(t) + Kp(t) * error(t)`

**Phase-Dependent Gains:**

| Phase | Stance % | Default Kp | Rationale
|-----|-----
| Weight Acceptance | 0-20% | 0.8 | Support loading response
| Mid-Stance | 20-60% | 0.4 | Allow natural weight transfer
| Push-Off | 60-100% | 0.8 | Enhance propulsion


The error inherently encodes impairment severity—larger deficits produce larger corrections without explicit severity multiplication.

## Practical Examples

```python
# Batch analysis
for i in range(len(reader.meta_df)):
    sample = reader.get_sample(i)
    features = extractor.extract(sample)
    label, margin = clf.predict_row(sample["grf"])
    grf_corrected, metrics = corrector.apply_correction(sample["grf"], reference)
```

## Troubleshooting

| Issue | Solution
|-----|-----
| "No HC samples found" | Ensure metadata contains CLASS_LABEL = "HC"
| "Classifier not fitted" | Run training.py first
| Low accuracy | Adjust n_components (0.90-0.99) or svm_C (0.1-10)


## Future Enhancements

- Real-time GRF streaming with force plate hardware
- Multi-joint analysis (knee, hip moments)
- Deep learning classifier (CNN/LSTM)
- Subject-specific online adaptation
- ROS/CAN bus integration for exoskeleton control
- Bilateral limb comparison
- Clinical PDF report generation


## References

1. Perry & Burnfield (2010). *Gait Analysis: Normal and Pathological Function*
2. Robinson et al. (1987). *J. Manipulative Physiol. Ther.*, 10(4), 172-176
3. Kirtley (2006). *Clinical Gait Analysis: Theory and Practice*
4. Banala et al. (2009). *IEEE Trans. Neural Syst. Rehabil. Eng.*, 17(1), 2-8
5. Shi et al. (2019). *Chinese J. Mech. Eng.*, 32(1), 74


## License

MIT License
