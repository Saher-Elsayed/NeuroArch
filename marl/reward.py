"""
Comfort-augmented reward function (Eq. 12, Paper Section VI-B).

r_t = -lambda_E * sum(E_z_norm)
    + lambda_C * sum(kappa_SNN * 1[PMV in [-0.5,+0.5]])
    - lambda_P * sum(PPD_norm)

All terms dimensionless; E and PPD normalised by rule-based reference.
"""
import torch


class ComfortAugmentedReward:
    def __init__(self, lambda_E=1.0, lambda_C=2.0, lambda_P=0.5,
                 E_ref=None, n_zones=6):
        self.lE = lambda_E
        self.lC = lambda_C
        self.lP = lambda_P
        self.E_ref = E_ref      # per-building reference (set at init)
        self.n_zones = n_zones

    def set_reference(self, E_ref: float):
        """Set building-specific rule-based energy reference (kWh/m2/yr)."""
        self.E_ref = E_ref

    def __call__(self, energy_kwh, kappa_snn, pmv_values, ppd_values):
        """
        Args (all per-zone tensors, shape (n_zones,)):
            energy_kwh: zone energy consumption this timestep
            kappa_snn:  SNN comfort confidence in [0,1]
            pmv_values: EnergyPlus PMV per zone
            ppd_values: EnergyPlus PPD per zone (0-100)
        Returns:
            scalar reward
        """
        assert self.E_ref is not None, "Call set_reference() first"
        E_norm   = energy_kwh / self.E_ref
        in_band  = ((pmv_values >= -0.5) & (pmv_values <= 0.5)).float()
        PPD_norm = ppd_values / 100.0
        r = (-self.lE * E_norm.sum()
             + self.lC * (kappa_snn * in_band).sum()
             - self.lP * PPD_norm.sum())
        return r.item()
