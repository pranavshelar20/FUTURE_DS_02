"""Step 3: inject results into the HTML template and write docs/index.html."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
t=(ROOT/'src/templates/dashboard_template.html').read_text()
core=(ROOT/'outputs/results_core.json').read_text()
extra=(ROOT/'outputs/results_cohorts.json').read_text()
(ROOT/'docs/index.html').write_text(t.replace('__DATA__',core).replace('__EXTRA__',extra))
print('Dashboard written to docs/index.html')
