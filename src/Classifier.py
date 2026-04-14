class Classifier:

    def classify(self, features): # me7tageen n2aked el thresholds

        heel = features["heel_peak"]
        toe = features["toe_peak"]
        hip = features["hip_range"]

        if hip < 20 and heel < 0.6:
            return "Hip + Knee Impairment"

        elif hip < 20:
            return "Hip Impairment"

        elif heel < 0.6 or toe < 0.6:
            return "Knee Impairment"

        else:
            return "Normal"