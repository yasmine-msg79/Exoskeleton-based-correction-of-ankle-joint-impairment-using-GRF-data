class Correction:
    

    def suggest(self, label):

        if label == "Hip Impairment":
            return "Increase hip flexion using exoskeleton assistance"

        elif label == "Knee Impairment":
            return "Stabilize stance phase and increase push-off"

        elif label == "Hip + Knee Impairment":
            return "Assist hip flexion and improve foot loading"

        else:
            return "No correction needed"

    def apply_correction(self, signals, label):

        hip_measured = signals["hip_measured"].copy()
        hip_reference = signals["hip_reference"].copy()
        error = hip_reference - hip_measured

        if label in ["Hip Impairment", "Hip + Knee Impairment"]:
            hip_assisted = hip_measured + error * 1.3

        return hip_assisted
