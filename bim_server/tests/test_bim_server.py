"""Unit tests for BIM server delta logic."""
import pytest
from bim_server.bim_server import BIMServer
from bim_server.websocket_delta import DifferentialEncoder


def test_delta_returns_none_if_unchanged():
    srv = BIMServer(ifc_path="nonexistent.ifc")
    d1 = srv.update_comfort("zone_1", "Neutral", 0.9, 0.1, 6.0)
    d2 = srv.update_comfort("zone_1", "Neutral", 0.9, 0.1, 6.0)
    assert d1 is not None
    assert d2 is None  # unchanged -> no delta


def test_delta_triggers_on_class_change():
    srv = BIMServer(ifc_path="nonexistent.ifc")
    srv.update_comfort("zone_1", "Neutral", 0.9, 0.1, 6.0)
    d = srv.update_comfort("zone_1", "Warm", 0.85, 0.6, 9.0)
    assert d is not None
    assert d["props"]["ComfortClass"] == "Warm"


def test_hvac_delta():
    srv = BIMServer(ifc_path="nonexistent.ifc")
    d = srv.update_hvac("zone_2", 22.0, 0.3, 0.7, 500, 0.88)
    assert d["pset"] == "Pset_HVACConfig"
    assert d["props"]["SupplyTempSet"] == 22.0


def test_differential_encoder_no_repeat():
    enc = DifferentialEncoder()
    b1 = enc.encode("GUID-001", "Pset_ThermalComfort", {"ComfortClass": "Neutral"})
    b2 = enc.encode("GUID-001", "Pset_ThermalComfort", {"ComfortClass": "Neutral"})
    assert b1 is not None
    assert b2 is None


def test_differential_encoder_different_zones():
    enc = DifferentialEncoder()
    b1 = enc.encode("GUID-001", "Pset_ThermalComfort", {"ComfortClass": "Neutral"})
    b2 = enc.encode("GUID-002", "Pset_ThermalComfort", {"ComfortClass": "Neutral"})
    assert b1 is not None and b2 is not None  # different GUIDs
