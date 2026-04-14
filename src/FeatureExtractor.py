import numpy as np
class FeatureExtractor:


    def extract(self, signals):

        heel_peak = np.max(signals["heel"])
        toe_peak = np.max(signals["toe"])

        hip_range = np.max(signals["hip_measured"]) - np.min(signals["hip_measured"])

        return {
            "heel_peak": heel_peak,
            "toe_peak": toe_peak,
            "hip_range": hip_range
        }