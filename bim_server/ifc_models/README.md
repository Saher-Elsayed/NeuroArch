# IFC Model Files

Place IFC 4.3 files here. Not included in repo due to proprietary BIM data.

## Expected Files
- `medium_office.ifc` — Medium Office Reference Building (4,982 m², 6 zones)
- `residential.ifc`   — Residential Reference Building (3,135 m², 4 zones)
- `mixed_use.ifc`     — Mixed-Use Reference Building (8,210 m², 8 zones)

## Generating Test IFC
```python
import ifcopenshell, ifcopenshell.api
model = ifcopenshell.api.run("project.create_file")
# see bim_server/bim_server.py for Pset schema
```

## Pset Schema
See `bim_server/ifc_pset_schema.json` for property set definitions.
