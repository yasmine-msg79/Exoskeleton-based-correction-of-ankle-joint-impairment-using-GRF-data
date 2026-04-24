import numpy as np
import pandas as pd

class SignalReader:

    def __init__(self, grf_file, metadata_file):
        self.grf_file = grf_file
        self.metadata_file = metadata_file

        self.grf_df = pd.read_csv(grf_file)
        self.meta_df = pd.read_csv(metadata_file)

    def get_sample(self, index):

        row = self.grf_df.iloc[index]
        label = self.meta_df.iloc[index]["CLASS_LABEL"]
        Affected_Limb = self.meta_df.iloc[index]["AFFECTED_SIDE"]
        # handling Nans
        if pd.isna(Affected_Limb):
            Affected_Limb = -1   
        else:
            Affected_Limb = int(Affected_Limb)
            
        SIDE_MAP = {
        -1: "none",
        0: "left",
        1: "right",
        2: "both"
        }
        
        side_label = SIDE_MAP[Affected_Limb]

        grf = row[3:].values.astype(float)
        t = np.linspace(0, 1, len(grf))

        return {
            "time": t,
            "grf": grf,
            "label": label,
            "Affected_Limb": side_label
        }