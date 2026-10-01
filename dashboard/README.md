# Dashboard (deferred)

Deliberately not built yet (Stage 6). Everything a dashboard would show is already
produced by `neurotune report`: `report.md`, `summary.json`, `trials.csv` and
`convergence.png`. A future dashboard should read only those files or the SQLite
database, so it can never display a result that the CLI cannot reproduce.
