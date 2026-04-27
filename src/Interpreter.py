import pandas as pd
import SignalReader as sigR
class Interpreter:

    def interpret(self, features, label, index):
        meta_path = "D:\Repos\Gait-Analysis-For-Hip-Exo-Control\Data\GRF_metadata.csv"
        meta_df = pd.read_csv(meta_path)
        if label == "normal" or meta_df["CLASS_LABEL"].iloc[index] == "HC":
            return "normal"

        # first_peak = features["first_peak"]
        # second_peak = features["second_peak"]

        # if first_peak <= 0:
        #     return "impaired"

        # push_ratio = second_peak / first_peak

        # if push_ratio < 0.80:
        #     return "weak_push_off"

        return "impaired"