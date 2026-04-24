class ReferenceBuilder:

    def build(self, reader):

        hc_signals = []

        meta = reader.meta_df

        for i in range(len(meta)):

            label = str(meta.iloc[i]["CLASS_LABEL"]).strip()

            if label == "HC":
                sample = reader.get_sample(i)
                hc_signals.append(sample["grf"])

        if len(hc_signals) == 0:
            raise ValueError("No HC samples found for reference")

        return sum(hc_signals) / len(hc_signals)