# Unreal Engine 5 Content

## Required UE5 Plugins
- WebSockets (built-in)
- IWebSocket (built-in)
- Procedural Mesh Component

## Material Parameter Collection: MPC_ComfortColors
Scalar parameters per zone (1-6):
- `Zone{N}_ComfortIndex`: 0=Cold, 1=Cool, 2=Neutral, 3=Warm, 4=Hot
- `Zone{N}_Saturation`: kappa_SNN [0,1]; <0.6 fades to gray (uncertain)
- `Zone{N}_ForecastRibbon`: 0-1 forecast confidence for 30-min horizon

## Blueprints
- `BP_ComfortHeatmap`: drives zone material from BIMWebSocketClient
- `BP_MARLPolicyOverlay`: real-time policy visualization (Section IX-C)
- `BP_ForecastRibbon`: 30-min thermal forecast ribbon (Section V-B)

## IFC Models
Place `.ifc` files in `bim_server/ifc_models/`:
- `medium_office.ifc`  (4,982 m², 6 zones, 15 HVAC zones)
- `residential.ifc`    (3,135 m², 4 zones)
- `mixed_use.ifc`      (8,210 m², 8 zones)
