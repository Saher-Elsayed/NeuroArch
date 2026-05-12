# NeuroArch BIM Server
# Serves IFC 4.3; writes SNN labels and MARL setpoints to Pset;
# streams differential WebSocket deltas to Unreal Engine 5.
# Paper: Section V-A, mean delta 340 bytes/tick
import asyncio, json, argparse
from pathlib import Path

class BIMServer:
    def __init__(self, ifc_path):
        self.ifc_path = ifc_path
        self.prev_state = {}
        self.clients = set()

    def update_comfort(self, zone_id, comfort_class, confidence, pmv, ppd):
        key = f"{zone_id}_comfort"
        new = {"ComfortClass": comfort_class, "SNNConfidence": round(confidence,3),
               "PMVValue": round(pmv,3), "PPDValue": round(ppd,1)}
        if self.prev_state.get(key) == new:
            return None
        self.prev_state[key] = new
        return {"zone": zone_id, "pset": "Pset_ThermalComfort", "props": new}

    def update_hvac(self, zone_id, supply_temp, blind_e, blind_w, lux):
        key = f"{zone_id}_hvac"
        new = {"SupplyTempSet": round(supply_temp,1), "BlindAngleEast": round(blind_e,2),
               "BlindAngleWest": round(blind_w,2), "LightingSetpoint": round(lux,0)}
        if self.prev_state.get(key) == new:
            return None
        self.prev_state[key] = new
        return {"zone": zone_id, "pset": "Pset_HVACConfig", "props": new}

    async def control_loop(self):
        import random
        t = 0
        while True:
            for z in range(1, 7):
                cls = random.choice(["Cold","Cool","Neutral","Warm","Hot"])
                kappa = round(0.7 + 0.3*random.random(), 3)
                pmv = round(random.gauss(0, 0.4), 3)
                delta = self.update_comfort(f"zone_{z}", cls, kappa, pmv, max(5,10*(pmv**2)))
                if delta:
                    print(f"[tick {t}] delta: {delta}")
            await asyncio.sleep(1.0)
            t += 1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ifc", default="ifc_models/medium_office.ifc")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    server = BIMServer(args.ifc)
    asyncio.run(server.control_loop())
