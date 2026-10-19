"""Loaders for the Surabaya case study (data/ folder of this repository)."""
from __future__ import annotations

import csv
from pathlib import Path


def load_feeders(path):
    """Feeder table of Akhra, Vanany and Arief (masked units).

    Returns two dicts keyed by feeder name: (mean, sd, rating) and the reported
    reliability p_j."""
    stations, reliability = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name = r["feeder"]
            stations[name] = (int(r["demand_mean"]), int(r["demand_sd"]), int(r["rating"]))
            reliability[name] = float(r["reported_reliability"])
    return stations, reliability


def load_surabaya(data_dir):
    """Coordinates, demand, population and motorcycle travel times of the case study,
    keyed by the feeder names of feeders.csv and the sub-district names of demand.csv."""
    data_dir = Path(data_dir)
    with open(data_dir / "feeders.csv", newline="") as f:
        alias = {r["feeder"]: r["station"] for r in csv.DictReader(f)}
    with open(data_dir / "stations.csv", newline="") as f:
        st = {r["station"]: r for r in csv.DictReader(f)}
    with open(data_dir / "demand.csv", newline="") as f:
        dem = list(csv.DictReader(f))
    minutes = {j: {} for j in alias}
    back = {v: k for k, v in alias.items()}
    with open(data_dir / "travel_time.csv", newline="") as f:
        for r in csv.DictReader(f):
            minutes[back[r["station"]]][r["subdistrict"]] = float(r["minutes_motorcycle"])
    return dict(
        feeders=list(alias),
        feeder_lat={j: float(st[s]["lat"]) for j, s in alias.items()},
        feeder_lon={j: float(st[s]["lon"]) for j, s in alias.items()},
        subdistricts=[r["subdistrict"] for r in dem],
        sub_lat={r["subdistrict"]: float(r["lat"]) for r in dem},
        sub_lon={r["subdistrict"]: float(r["lon"]) for r in dem},
        population={r["subdistrict"]: int(r["population"]) for r in dem},
        demand={r["subdistrict"]: int(r["demand_motorcycle"]) + int(r["demand_car"]) for r in dem},
        minutes=minutes,
    )
