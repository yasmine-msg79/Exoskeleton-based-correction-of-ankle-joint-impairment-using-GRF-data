import numpy as np

class SignalSimulator:

    def __init__(self, duration=5, sampling_rate=100):
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.t = np.linspace(0, duration, duration * sampling_rate)

    def simulate(self, heel_peak, toe_peak, hip_range):

        heel = heel_peak * np.maximum(0, np.sin(2 * np.pi * 1 * self.t))
        toe = toe_peak * np.maximum(0, np.sin(2 * np.pi * 1 * self.t + np.pi / 2))

        hip = hip_range/2 * np.sin(2 * np.pi * 1 * self.t)
        
        normal_range = 30  # normal hip range of motion in degrees
        hip_reference = normal_range/2 * np.sin(2 * np.pi * 1 * self.t)

        return {
            "time": self.t,
            "heel": heel,
            "toe": toe,
            "hip_measured": hip,
            "hip_reference": hip_reference
        }