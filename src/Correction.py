from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

# ── Tuneable constants ────────────────────────────────────────────────────────
# Fraction of the pointwise error e(t) to feed back each phase.
# 1.0 = fully close the gap in one step; 0.5 = correct half the error, etc.
KP_LOADING  = 0.8   # weight-acceptance phase  (0–20% stance)
KP_MID      = 0.4   # mid-stance               (20–60% stance)
KP_PUSHOFF  = 0.8   # push-off / pre-swing     (60–100% stance)

# RMSE value (in BW units) that maps to severity = 1.0
SEVERITY_THRESHOLD = 0.3


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class Correction:
    """
    Computes GRF error metrics against a healthy-mean reference and applies
    a phase-dependent Assist-As-Needed correction to the measured waveform.
    """

    def compute_error(
        self,
        grf_measured: np.ndarray,
        grf_reference: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute quantified error metrics between the patient's GRF waveform
        and the healthy reference.

        Parameters
        ----------
        grf_measured  : 1-D array of length N, BW-normalised vertical GRF
        grf_reference : 1-D array of same length, healthy-mean GRF

        Returns
        -------
        dict with keys:
            rmse           — Root Mean Square Error  [BW]
            mae            — Mean Absolute Error     [BW]
            delta_f1       — First-peak error        [BW]   (ref − measured)
            delta_f2       — Second-peak error       [BW]   (ref − measured)
            delta_impulse  — Impulse deficit         [BW·s] (ref − measured)
            si_f1          — Symmetry index for first peak   [%]
            si_f2          — Symmetry index for second peak  [%]
            severity       — Scalar 0–1 derived from RMSE
        """
        grf_m = np.asarray(grf_measured,  dtype=np.float64).ravel()
        grf_r = np.asarray(grf_reference, dtype=np.float64).ravel()

        # Align lengths
        n = min(len(grf_m), len(grf_r))
        grf_m, grf_r = grf_m[:n], grf_r[:n]
        mid = n // 2

        # ── Waveform-level metrics ────────────────────────────────────────────
        e = grf_r - grf_m                                      # pointwise error
        rmse = float(np.sqrt(np.mean(e ** 2)))
        mae  = float(np.mean(np.abs(e)))

        # ── Peak errors   ──────────────────────────────
        f1_ref  = float(np.max(grf_r[:mid]))
        f1_meas = float(np.max(grf_m[:mid]))
        f2_ref  = float(np.max(grf_r[mid:]))
        f2_meas = float(np.max(grf_m[mid:]))

        delta_f1 = f1_ref  - f1_meas
        delta_f2 = f2_ref  - f2_meas

        # Symmetry Index (%)
        si_f1 = (
            200.0 * abs(delta_f1) / (f1_ref + f1_meas)
            if (f1_ref + f1_meas) > 0 else 0.0
        )
        si_f2 = (
            200.0 * abs(delta_f2) / (f2_ref + f2_meas)
            if (f2_ref + f2_meas) > 0 else 0.0
        )

        # ── Impulse deficit ───────────────────────────────────────────────────
        delta_impulse = float(np.trapezoid(grf_r) - np.trapezoid(grf_m))

        # ── Severity scalar  ────────────────────────────────────
        severity = _clamp(rmse / SEVERITY_THRESHOLD)

        return {
            "rmse":          rmse,
            "mae":           mae,
            "delta_f1":      delta_f1,
            "delta_f2":      delta_f2,
            "delta_impulse": delta_impulse,
            "si_f1":         si_f1,
            "si_f2":         si_f2,
            "severity":      severity,
        }

    # ── Correction formula ────────────────────────────────────────────────────

    def apply_correction(
        self,
        grf_measured:  np.ndarray,
        grf_reference: np.ndarray,
        kp_loading:  float = KP_LOADING,
        kp_mid:      float = KP_MID,
        kp_pushoff:  float = KP_PUSHOFF,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply a phase-dependent Assist-As-Needed correction.

        Formula:
            grf_corrected(t) = grf_measured(t) + Kp(t) · e(t)
            e(t) = grf_reference(t) − grf_measured(t)
            Kp(t) = phase_gain

        e(t) is proportional to the patient's impairment magnitude;
        doubling-down with severity would nearly zero-out corrections for
        mild-to-moderate patients.

        Parameters
        ----------
        grf_measured  : patient's raw GRF waveform (1-D, BW-normalised)
        grf_reference : healthy-mean reference waveform
        kp_loading    : fraction of error corrected in weight-acceptance (0–20%)
        kp_mid        : fraction of error corrected in mid-stance (20–60%)
        kp_pushoff    : fraction of error corrected in push-off (60–100%)

        Returns
        -------
        grf_corrected : corrected waveform (same length as input)
        info          : dict with error metrics + applied gains
        """
        grf_m = np.asarray(grf_measured,  dtype=np.float64).ravel().copy()
        grf_r = np.asarray(grf_reference, dtype=np.float64).ravel()

        n = min(len(grf_m), len(grf_r))
        grf_m, grf_r = grf_m[:n], grf_r[:n]

        # Error metrics (includes severity for reporting)
        metrics = self.compute_error(grf_m, grf_r)
        e = grf_r - grf_m                                      # pointwise error

        # ── Build phase-dependent Kp mask ─────────────────────────────────────
        # Kp is the direct fraction of e(t) to feed back — no severity scaling.
        pct = np.linspace(0, 100, n, endpoint=False)

        Kp = np.where(
            pct < 20,                                          # weight-acceptance
            kp_loading,
            np.where(
                pct < 60,                                      # mid-stance
                kp_mid,
                kp_pushoff,                                    # push-off
            ),
        )

        # ── Apply correction ──────────────────────────────────────────────────
        grf_corrected = grf_m + Kp * e

        info = {
            **metrics,
            "kp_loading_applied":  float(kp_loading),
            "kp_mid_applied":      float(kp_mid),
            "kp_pushoff_applied":  float(kp_pushoff),
        }
        return grf_corrected, info

    # ── Textual suggestions ───────────────────────────────────────────────────

    def suggest(
        self,
        error_metrics: Dict[str, float],
        class_label: str = "impaired",
    ) -> str:
        """
        Generate clinically meaningful correction suggestions based on quantified
        error metrics .

        Parameters
        ----------
        error_metrics : dict returned by compute_error()
        class_label   : 'normal' or 'impaired' (or raw dataset CLASS_LABEL)

        Returns
        -------
        Multi-line suggestion string
        """
        label_str = str(class_label).strip().upper()
        if label_str == "HC" or class_label == "normal":
            return "[OK] Gait within normal limits - no correction required."

        lines = []
        rmse     = error_metrics.get("rmse", 0.0)
        delta_f1 = error_metrics.get("delta_f1", 0.0)
        delta_f2 = error_metrics.get("delta_f2", 0.0)
        d_imp    = error_metrics.get("delta_impulse", 0.0)
        si_f1    = error_metrics.get("si_f1", 0.0)
        si_f2    = error_metrics.get("si_f2", 0.0)
        severity = error_metrics.get("severity", 0.0)

        # Overall severity
        if severity > 0.66:
            lines.append("[!!] Severe gait impairment detected.")
        elif severity > 0.33:
            lines.append("[!]  Moderate gait impairment detected.")
        else:
            lines.append("[i]  Mild gait impairment detected.")

        # First-peak / weight-acceptance
        if delta_f1 > 0.05:
            lines.append(
                f"  - Weight-acceptance deficit: dF1 = {delta_f1:+.3f} BW  "
                f"(SI = {si_f1:.1f}%) -> increase loading-phase hip extension assistance."
            )
        elif delta_f1 < -0.05:
            lines.append(
                f"  - Weight-acceptance excess: dF1 = {delta_f1:+.3f} BW -> "
                "reduce early-stance loading."
            )

        # Second-peak / push-off
        if delta_f2 > 0.05:
            lines.append(
                f"  - Push-off deficit: dF2 = {delta_f2:+.3f} BW  "
                f"(SI = {si_f2:.1f}%) -> increase push-off / pre-swing hip flexion assistance."
            )
        elif delta_f2 < -0.05:
            lines.append(
                f"  - Push-off excess: dF2 = {delta_f2:+.3f} BW -> "
                "reduce terminal-stance assistance."
            )

        # Impulse
        if d_imp > 0.5:
            lines.append(
                f"  - Overall loading deficit: dImpulse = {d_imp:+.2f} BW*s -> "
                "increase full-cycle hip assistance."
            )

        # RMSE summary
        lines.append(f"  RMSE = {rmse:.4f} BW  |  Severity = {severity:.2f}")
        return "\n".join(lines)