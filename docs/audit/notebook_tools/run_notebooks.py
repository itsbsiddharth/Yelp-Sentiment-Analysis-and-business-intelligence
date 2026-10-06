"""Execute copies of the original notebooks and record per-cell status.

Modes (every edit is logged in OUTDIR/summary.json):
  verbatim  only `!pip install` lines are removed (they would install into whatever `pip` is on PATH,
            not the kernel's environment); everything else runs as written.
  patched   additionally rewrites the hard-coded Windows data path to data/ for reads and out/ for
            the EDA notebook's to_csv calls, so the inputs are never overwritten.
WORKDIR is the kernel's working directory: put a *copy* of data/ there, plus Classification_py.py
(the reference repo's Classification.py) for the patched run.

Usage: python run_notebooks.py MODE OUTDIR WORKDIR NB [NB...]
"""
import sys, json, time, copy, re, os
import nbformat
from nbclient import NotebookClient

mode, outdir, workdir = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = mode
nbs = sys.argv[4:]
os.makedirs(outdir, exist_ok=True)
WIN_DATA = re.compile(r"""r?(['"])C:[/\\]Users[/\\]besid[/\\]OneDrive[/\\]Desktop[/\\]on campus internship[/\\]machine learning[/\\]yelpproject[/\\]data[/\\]""")

def patch(src, log, idx):
    new = src
    if MODE == 'patched':
        if 'to_csv' in new:
            new = WIN_DATA.sub(r"\1out/", new)      # nb1 write location -> out/ (never the inputs)
        else:
            new = WIN_DATA.sub(r"\1data/", new)     # reads -> data/
    new = re.sub(r"^(\s*)!pip install.*$", r"\1pass  # [audit] removed: pip install", new, flags=re.M)
    if new != src:
        log.append({'cell': idx, 'before': src, 'after': new})
    return new

summary = {}
for path in nbs:
    name = os.path.basename(path).strip()
    nb = nbformat.read(path, as_version=4)
    log = []
    if True:  # both modes strip `!pip install`; only 'patched' rewrites paths
        for i, c in enumerate(nb.cells):
            if c.cell_type == 'code':
                c.source = patch(c.source, log, i)
    t0 = time.time()
    client = NotebookClient(nb, timeout=7200, kernel_name='python3', allow_errors=True,
                            resources={'metadata': {'path': workdir}})
    try:
        client.execute()
        status = 'completed'
    except Exception as e:
        status = f'aborted: {type(e).__name__}: {e}'[:500]
    cells = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != 'code':
            continue
        errs = [o for o in c.get('outputs', []) if o.get('output_type') == 'error']
        cells.append({'cell': i, 'error': (f"{errs[0]['ename']}: {errs[0]['evalue']}"[:300] if errs else None)})
    out_nb = os.path.join(outdir, name)
    nbformat.write(nb, out_nb)
    summary[name] = {'status': status, 'seconds': round(time.time() - t0, 1), 'patches': log,
                     'errors': [c for c in cells if c['error']], 'n_code_cells': len(cells)}
    print(name, status, summary[name]['seconds'], 's; errors:', len(summary[name]['errors']), flush=True)
json.dump(summary, open(os.path.join(outdir, 'summary.json'), 'w'), indent=1)
