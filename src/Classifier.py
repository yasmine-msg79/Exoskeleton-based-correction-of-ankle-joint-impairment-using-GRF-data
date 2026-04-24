# class Classifier:

#     def classify(self, features): # me7tageen n2aked el thresholds

#         heel = features["heel_peak"]
#         toe = features["toe_peak"]
#         hip = features["hip_range"]
#         labels = []

#             # --- HIP RULES ---
#         if hip < 15:
#             labels.append("Severe hip mobility limitation")
#         elif hip < 25:
#             labels.append("Mild hip mobility limitation")

#         # --- HEEL RULES ---
#         if heel < 0.4:
#             labels.append("Reduced heel strike loading")

#         # --- TOE RULES ---
#         if toe < 0.4:
#             labels.append("Reduced push-off force")

#         # --- COMBINATION RULES (IMPORTANT) ---

#         if hip < 15 and heel < 0.4:
#             labels.append("Stance phase + hip mobility deficit")

#         if heel < 0.4 and toe < 0.4:
#             labels.append("Global gait force reduction")

#         if len(labels) == 0:
#             labels.append("Normal")
#         return labels
