"""
ASHRAE Guideline 36 rule-based baseline controller.
Cooling setpoint 24°C, heating 21°C, minimum airflow 0.30, no economizer.
Paper: Section VII (Baseline Controllers)
"""


class RuleBasedG36:
    """ASHRAE Guideline 36 baseline (fixed setpoints, no demand response)."""
    COOLING_SP = 24.0    # degC
    HEATING_SP = 21.0    # degC
    MIN_AIRFLOW = 0.30   # fraction
    LUX_DEFAULT = 500.0  # lux

    def __init__(self, schedule_start=8, schedule_end=18):
        self.start = schedule_start
        self.end   = schedule_end

    def act(self, obs: dict) -> dict:
        """
        obs: dict with keys 'T_zone', 'hour_of_day', 'occupancy'
        returns: dict with setpoints
        """
        occupied = obs.get("occupancy", 1) and self.start <= obs.get("hour_of_day", 12) <= self.end
        T_zone = obs.get("T_zone", 22.0)
        T_sup  = (self.COOLING_SP if T_zone > self.COOLING_SP else
                  self.HEATING_SP if T_zone < self.HEATING_SP else T_zone)
        return {
            "T_supply_C": T_sup,
            "blind_east": 0.5, "blind_west": 0.5,
            "lighting_lux": self.LUX_DEFAULT if occupied else 0.0,
            "airflow_fraction": self.MIN_AIRFLOW if occupied else 0.10,
        }
