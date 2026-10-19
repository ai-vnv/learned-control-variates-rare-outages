PY ?= python3

.PHONY: install test boundaries experiments figures gif numbers all clean-figures

install:            ## install the lcv package and its dependencies
	$(PY) -m pip install -e ".[test]"

test:               ## fast checks of the theorems and estimators (seconds)
	$(PY) -m pytest -q

boundaries:         ## download the GADM sub-district boundaries used by the maps
	$(PY) scripts/fetch_boundaries.py

experiments:        ## rerun every simulation (about 1 hour on 2 cores); overwrites results/
	$(PY) scripts/run_experiments.py      # ~40 min: Fig. 1(a,c), Table I, checks in V-B
	$(PY) scripts/run_running_multi.py 400  # ~4 min: Fig. 1(b)
	$(PY) scripts/run_surabaya.py         # ~8 min: Embong Wungu reserve decision (V-C)
	$(PY) scripts/run_map.py              # ~2 min: Fig. 2 and Table II
	$(PY) scripts/run_relerr.py           # ~2 min: animated map

figures: boundaries ## rebuild figures/ and tables/ from results/ (seconds)
	$(PY) scripts/make_figures.py
	$(PY) scripts/make_tables.py
	$(PY) scripts/make_map.py
	$(PY) scripts/make_surabaya.py

gif: boundaries     ## animated map of the estimated outage probability
	$(PY) scripts/make_gif.py

numbers:            ## print every number quoted in the paper; writes results/paper_numbers.md
	$(PY) scripts/report_numbers.py

all: experiments figures gif numbers
