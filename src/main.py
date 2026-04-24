# import numpy as np
# from SignalReader import SignalSimulator
# from FeatureExtractor import FeatureExtractor
# from PhaseDetector import PhaseDetector
# from Classifier import Classifier
# from Correction import Correction


# class main:

#     def __init__(self):

#         self.simulator = SignalSimulator()
#         self.extractor = FeatureExtractor()
#         self.phase_detector = PhaseDetector()
#         self.classifier = Classifier()
#         self.controller = Correction()

#     def run(self, heel_peak, toe_peak, hip_range):

#         signals = self.simulator.simulate(heel_peak, toe_peak, hip_range)

#         features = self.extractor.extract(signals)

#         phase_ranges = self.phase_detector.detectPhase(signals)
        
#         phase_states = self.phase_detector.getPhasestates()

#         labels = self.classifier.classify(features)

#         corrections, hip_related = self.controller.suggest(labels)

#         corrected_hip = self.controller.apply_correction(signals, hip_related)

#         results = {
#             "signals": signals, # the 3 original signals (heel, toe, hip)
#             "features": features, # the extracted features from the signals
#             "phases": phase_ranges, # the detected gait phases (stance, swing, heel strike, toe off)
#             "phase_states": phase_states, # the states of each gait phase
#             "classification": labels, # verdict (normal, knee impairmant, hip impairment)
#             "correction": corrections, # the suggested correction based on the classification
#             "corrected_hip": corrected_hip # corrected hip signal after applying the suggested correction
#         }

#         return results