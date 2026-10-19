# Data

| File | Content | Source |
|---|---|---|
| `feeders.csv` | The 11 Surabaya feeders: masked daily peak-load mean and standard deviation, rating, and reported reliability `p_j` (Table I of [8]); `station` gives the matching name in the other files | Akhra, Vanany and Arief, arXiv:2307.05470 |
| `stations.csv`, `demand.csv`, `travel_time.csv`, `params.json` | Feeder coordinates, the 31 sub-districts (population, EV charging demand), and travel times | [ai-vnv/rare-event-ev-charging](https://github.com/ai-vnv/rare-event-ev-charging) (MIT licence) |
| `basemap/` | Carto *Positron* render of the map window | Map data (c) OpenStreetMap contributors (ODbL); style (c) Carto (CC BY 3.0); copied from ai-vnv/rare-event-ev-charging |
| `boundaries/` (not committed) | GADM 4.1 level-3 sub-district polygons, written by `scripts/fetch_boundaries.py` | [GADM](https://gadm.org) via the npm package geojson-indonesia 1.0.1; GADM does not allow redistribution, so the file is downloaded on demand |

The load and rating values are masked and rescaled by the utility. Coordinates in
`stations.csv` and `demand.csv` are rounded to six decimals, which changes nothing in the
results.
