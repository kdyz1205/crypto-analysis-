"""Systematic debug audit — checks everything."""
import ast, os, re, subprocess, json

bugs = []

# 1. Python syntax
py_files = []
for root, dirs, files in os.walk('server'):
    for f in files:
        if f.endswith('.py'): py_files.append(os.path.join(root, f))
py_files += [f for f in ['run.py','run_all.py','run_evolution.py','run_tg_bot.py'] if os.path.exists(f)]
for f in py_files:
    try: ast.parse(open(f, encoding='utf-8').read())
    except SyntaxError as e: bugs.append(f'SYNTAX {f}:{e.lineno}')

# 2. Imports
for mod in ['server.strategy.config','server.strategy.regime','server.strategy.zones',
            'server.strategy.trendlines','server.strategy.signals','server.strategy.replay',
            'server.strategy.indicators','server.strategy.catalog','server.strategy.evolution',
            'server.strategy.sr_strategy','server.factors.factor_engine',
            'server.routers.runtime','server.routers.live_execution']:
    try: __import__(mod)
    except Exception as e: bugs.append(f'IMPORT {mod}: {e}')

# 3. Config
from server.strategy.config import StrategyConfig
cfg = StrategyConfig()
if cfg.break_close_count > 2: bugs.append(f'CONFIG break_close_count={cfg.break_close_count}')
if cfg.min_rr_ratio > 3: bugs.append(f'CONFIG min_rr_ratio={cfg.min_rr_ratio}')

# 4. JS syntax (use shell redirect to avoid Windows encoding issues)
for f in ['frontend/js/execution/panel.js','frontend/js/workbench/chart.js','frontend/js/main.js',
          'frontend/js/workbench/decision_rail.js','frontend/js/services/runtime.js']:
    r = subprocess.run(f'node --input-type=module --check < "{f}"', shell=True, capture_output=True, text=True, timeout=10)
    if r.returncode != 0: bugs.append(f'JS_SYNTAX {f}')

# 5. HTML dead refs
html = open('frontend/v2.html', encoding='utf-8').read()
for did in ['v2-boot-status-slot','v2-glassbox','v2-research-btn','v2-chat-btn']:
    if did in html: bugs.append(f'HTML dead: {did}')

# 6. Server health
try:
    r = subprocess.run(['curl','-s','--max-time','3','http://127.0.0.1:8001/api/health'], capture_output=True, text=True, timeout=5)
    if 'ok' not in r.stdout: bugs.append('SERVER down')
except: bugs.append('SERVER unreachable')

# 7. Tests
r = subprocess.run(['python','-m','pytest','tests/execution/test_risk_rules.py','-q','--tb=no'], capture_output=True, text=True, timeout=15)
test_line = [l for l in r.stdout.split('\n') if 'passed' in l or 'failed' in l]

print(f'=== SYSTEMATIC DEBUG REPORT ===')
print(f'Python: {len(py_files)} files checked')
print(f'Bugs: {len(bugs)}')
for b in bugs: print(f'  {b}')
if test_line: print(f'Tests: {test_line[-1].strip()}')
if not bugs: print('ALL CLEAN')
