import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SignalReader import SignalReader
from FeatureExtractor import FeatureExtractor
from Interpreter import Interpreter
from ReferenceBuilder import ReferenceBuilder

# FILES (LEFT ONLY)

# INSERT YOUR PERSONAL PATHS HERE
# GRF_FILE = r"C:\Users\loq\Downloads\GRF_F_V_PRO_left.csv"
# META_FILE = r"C:\Users\loq\Downloads\GRF_metadata.csv"

GRF_FILE = r"D:\College\Year Four\Second Term\Rehab Project\Exoskeleton-based-correction-of-ankle-joint-impairment-using-GRF-data\\Data\\GRF_F_V_PRO_left.csv"
META_FILE = r"D:\College\Year Four\Second Term\Rehab Project\Exoskeleton-based-correction-of-ankle-joint-impairment-using-GRF-data\\Data\\GRF_metadata.csv"

# =====================================================
# INIT MODULES
# =====================================================
reader = SignalReader(GRF_FILE, META_FILE)
extractor = FeatureExtractor()
interpreter = Interpreter()

# =====================================================
# LOAD SAMPLE (WE WILL FORCE AN ANKLE CASE)
# =====================================================
def find_ankle_sample(reader):
    for i in range(len(reader.grf_df)):

        sample = reader.get_sample(i)
        label = sample["label"]   # already standardized

        if isinstance(label, str) and label.startswith("A"):
            return i

    return None


ankle_index = find_ankle_sample(reader)

if ankle_index is None:
    raise ValueError("No ankle sample found in dataset")

sample = reader.get_sample(ankle_index)

print("\nSelected sample index:", ankle_index)
print("Class label:", sample["label"])
print("Affected side:", sample["Affected_Limb"])

# =====================================================
# REFERENCE BUILDING (FROM HEALTHY ONLY)
# =====================================================
print("\nBuilding reference waveform from HC samples...")

ref_builder = ReferenceBuilder()
reference = ref_builder.build(reader)

# =====================================================
# FEATURE EXTRACTION
# =====================================================
features = extractor.extract(sample)

print("\nExtracted features:")
for k, v in features.items():
    print(f"{k}: {v:.4f}")

# =====================================================
# INTERPRETATION (FORCED ANKLE CASE)
# =====================================================
forced_label = "ankle_impairment"

result = interpreter.interpret(features, forced_label)

print("\nInterpretation result:", result)

# =====================================================
# PLOTTING: SIGNAL vs REFERENCE
# =====================================================
t = sample["time"]
grf = sample["grf"]

plt.figure()
plt.plot(t, grf, label="Ankle Impaired GRF")
plt.plot(t, reference, linestyle="--", label="Reference (Healthy Mean)")

plt.title("GRF Signal vs Reference")
plt.xlabel("Gait Cycle (%)")
plt.ylabel("Force (BW-normalized)")
plt.legend()
plt.grid()
plt.show()

# =====================================================
# FEATURE VISUAL DEBUG (OPTIONAL BUT USEFUL)
# =====================================================
plt.figure()
plt.bar(["First Peak", "Second Peak", "Impulse"],
        [features["first_peak"], features["second_peak"], features["impulse"]])

plt.title("Extracted Features")
plt.grid()
plt.show()