# class Correction:
#     def suggest(self, labels):

#         suggestions = []
#         hip_related = False

#         if "Severe hip mobility limitation" in labels:
#             suggestions.append("Increase hip assistance during swing phase")
#             hip_related = True

#         if "Mild hip mobility limitation" in labels:
#             suggestions.append("Slightly increase hip assistance during swing phase")
#             hip_related = True

#         if "Reduced heel strike loading" in labels:
#             suggestions.append("Increase heel strike assistance during stance phase")

#         if "Reduced push-off force" in labels:
#             suggestions.append("Increase push-off assistance during toe off")

#         if "Stance phase + hip mobility deficit" in labels:
#             suggestions.append("Increase hip assistance during stance phase")
#             hip_related = True

#         if "Global gait force reduction" in labels:
#             suggestions.append("Increase overall assistance during gait cycle")

#         if "Normal" in labels:
#             suggestions.append("No correction needed")

#         return suggestions, hip_related

#     def apply_correction(self, signals, hip_related):

#         hip_measured = signals["hip_measured"].copy()

#         if not hip_related:
#             return hip_measured

#         hip_reference = signals["hip_reference"].copy()

#         error = hip_reference - hip_measured

#         hip_assisted = hip_measured + 1.3 * error

#         return hip_assisted