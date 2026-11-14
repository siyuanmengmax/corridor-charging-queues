"""
Vehicle parameter profiles shared by both simulators and the dispatch rules.

  eu  battery N(70, 10^2) kWh, consumption 0.20 kWh/km: the synthetic corridor (close to the
      EU average new-BEV battery, IEA Global EV Outlook 2026; NREL 2023 sedan rate)
  us  battery N(90, 13^2) kWh, consumption 0.22 kWh/km: the I-95 case (US average new-BEV
      battery about 90 kWh in 2025, IEA Global EV Outlook 2026; NREL 2023 rates of
      300-430 Wh/mi for sedans and C/SUVs; range about 410 km = 255 mi, NREL 2025: 240-260 mi)

Phase 1 draws each vehicle's consumption from N(c, (0.1 c)^2); SUMO uses c for every vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleProfile:
    name: str
    battery_mean_kwh: float
    battery_std_kwh: float
    consumption_kwh_per_km: float

    @property
    def ref_energy_kwh(self) -> float:
        """Reference session of the port-status rules: 20-80% SOC of the mean battery."""
        return 0.6 * self.battery_mean_kwh

    def phase1_params(self) -> dict:
        return dict(battery_mean_kwh=self.battery_mean_kwh, battery_std_kwh=self.battery_std_kwh,
                    consumption_mean=self.consumption_kwh_per_km,
                    consumption_std=0.1 * self.consumption_kwh_per_km)


PROFILES = {
    "eu": VehicleProfile("eu", 70.0, 10.0, 0.20),
    "us": VehicleProfile("us", 90.0, 13.0, 0.22),
}


def get_profile(name: str = "eu") -> VehicleProfile:
    return PROFILES[name]
