# import numpy as np

# class PhaseDetector:
#     # flags for each phase based on signal thresholds
#     heel_strike_done = False
#     toe_off_done = False
#     mid_stance_phase_done = False
#     swing_phase_done = False

#     def detectPhase(self, signals):
        
        

#         heel_strike = np.any(signals["heel"] > 0.7 * np.max(signals["heel"]))
#         toe_off = np.any(signals["toe"] > 0.7 * np.max(signals["toe"]))
#         swing_phase = np.any(signals["hip_measured"] > 0)
#         mid_stance_phase = np.any(signals["hip_measured"] < 0)
        
#         # check if phases happened
#         if heel_strike:
#             self.heel_strike_done = True
#         if toe_off:
#             self.toe_off_done = True
#         if mid_stance_phase:
#             self.mid_stance_phase_done = True
#         if swing_phase:
#             self.swing_phase_done = True
            

#         return {
#             "heel_strike": heel_strike,
#             "toe_off": toe_off,
#             "mid_stance_phase": mid_stance_phase,
#             "swing_phase": swing_phase
#         }
        
#     def getPhasestates(self):
#         return {
#             "heel_strike_done": self.heel_strike_done,
#             "toe_off_done": self.toe_off_done,
#             "mid_stance_phase_done": self.mid_stance_phase_done,
#             "swing_phase_done": self.swing_phase_done
#         }