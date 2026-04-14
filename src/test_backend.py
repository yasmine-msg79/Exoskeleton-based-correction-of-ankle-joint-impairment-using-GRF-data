import matplotlib.pyplot as plt
from main import main as GaitPipeline

# create pipeline
pipeline = GaitPipeline()

# test parameters
heel_peak = 1.0  # hena threshold 0.6 lazem yeb2a aktar
toe_peak = 0.8 # hena threshold 0.6 lazem yeb2a aktar bardo
hip_range = 15   # hena threshold 20 lazem yeb2a a2al mn 20 3ashan yeb2a hip impairment

# run pipeline
results = pipeline.run(heel_peak, toe_peak, hip_range)

signals = results["signals"]
time = signals["time"]
heel = signals["heel"]
toe = signals["toe"]
hip = signals["hip_measured"]

corrected_hip = results["corrected_hip"]

# print backend outputs
print("Features:", results["features"])
print("Phases:", results["phases"])
print("Classification:", results["classification"])
print("Correction:", results["correction"])

# plot signals
plt.figure()
plt.plot(time, heel, label="Heel")
plt.plot(time, toe, label="Toe")
plt.plot(time, hip, label="Hip")
plt.legend()
plt.title("Simulated Gait Signals")
plt.show()

# plot correction
plt.figure()
plt.plot(time, hip, label="Original Hip")
plt.plot(time, corrected_hip, label="Corrected Hip")
plt.legend()
plt.title("Hip Correction")
plt.show()