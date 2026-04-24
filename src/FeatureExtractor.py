import numpy as np

class FeatureExtractor:

    def extract(self, signals):

        grf = signals["grf"]

        mid = len(grf) // 2

        first_peak = np.max(grf[:mid])
        second_peak = np.max(grf[mid:])

        impulse = np.trapezoid(grf)

        return {
            "first_peak": first_peak,
            "second_peak": second_peak,
            "impulse": impulse
        }