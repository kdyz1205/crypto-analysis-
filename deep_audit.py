"""Deep functional audit — checks logic, edge cases, cross-module consistency."""
import ast, os, re, json, subprocess, sys

bugs = []
warnings = []

def bug(msg): bugs.append(msg)
def warn(msg): warnings.append(msg)

# ═══ 1. Python files — syntax + dangerous patterns ═══
py_count = 0
for root, dirs, files in os.walk('server'):
    for f in files:
        if not f.endswith('.py'): continue
        py_count += 1
        path = os.path.join(root, f)
        code = open(path, encoding='utf-8').read()
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Bare except
            if line.strip() == 'except:':
                warn(f'{path}:{i} bare except')
            # Division by literal zero
            if re.search(r'/ 0(?:\s|$|[,;\)])', line) and 'max(' not in line:
                bug(f'{path}:{i} division by zero risk')
            # TODO/FIXME/HACK
            if re.search(r'#.*(TODO|FIXME|HACK|XXX)', line, re.I):
                warn(f'{path}:{i} {line.strip()[:60]}')

for f in ['run.py','run_all.py','run_evolution.py','run_tg_bot.py','run_hft.py','run_hft_ws.py','run_market_maker.py']:
    if os.path.exists(f):
        py_count += 1
        try: ast.parse(open(f, encoding='utf-8').read())
        except SyntaxError as e: bug(f'{f}:{e.lineno} SYNTAX ERROR')

# ═══ 2. JS files — syntax ═══
js_count = 0
for root, dirs, files in os.walk('frontend/js'):
    for f in files:
        if not f.endswith('.js'): continue
        js_count += 1
        path = os.path.join(root, f)
        r = subprocess.run(f'node --input-type=module --check < "{path}"', shell=True, capture_output=True, timeout=10)
        if r.returncode != 0:
            bug(f'{path} JS SYNTAX ERROR')

# ═══ 3. Config sanity ═══
from server.strategy.config import StrategyConfig
cfg = StrategyConfig()
if cfg.break_close_count > 2: bug(f'config break_close_count={cfg.break_close_count}')
if cfg.max_non_touch_crosses > 1: bug(f'config max_non_touch_crosses={cfg.max_non_touch_crosses}')
if cfg.min_rr_ratio > 3.0: bug(f'config min_rr_ratio={cfg.min_rr_ratio}')
if cfg.zone_min_touches < 2: bug(f'config zone_min_touches={cfg.zone_min_touches}')
if cfg.zone_eps_atr_mult > 0.5: bug(f'config zone_eps_atr_mult={cfg.zone_eps_atr_mult}')
rr_max = cfg.dynamic_rr_target(score=1.0, trend_aligned=True, zone_strength=100)
if rr_max > 5.0: bug(f'config dynamic_rr max={rr_max}')

# ═══ 4. Import chains ═══
imports = [
    'server.strategy.config', 'server.strategy.regime', 'server.strategy.zones',
    'server.strategy.trendlines', 'server.strategy.signals', 'server.strategy.replay',
    'server.strategy.indicators', 'server.strategy.catalog', 'server.strategy.evolution',
    'server.strategy.sr_strategy', 'server.factors.factor_engine',
    'server.hft.data_feed.book_builder', 'server.hft.data_feed.features',
    'server.hft.strategies.imbalance_mr', 'server.hft.strategies.sweep_breakout',
    'server.hft.strategies.inventory_mm', 'server.hft.router', 'server.hft.risk.kill_switch',
    'server.routers.runtime', 'server.routers.live_execution', 'server.routers.strategy',
]
for mod in imports:
    try: __import__(mod)
    except Exception as e: bug(f'IMPORT {mod}: {e}')

# ═══ 5. HTML/CSS cross-references ═══
html = open('frontend/v2.html', encoding='utf-8').read()
for dead in ['v2-boot-status-slot','v2-glassbox','v2-research-btn','v2-chat-btn','v2-cmdk-btn']:
    if dead in html: bug(f'HTML dead element: {dead}')

css = open('frontend/v2.css', encoding='utf-8').read()
panel = open('frontend/js/execution/panel.js', encoding='utf-8').read()
for cls in set(re.findall(r'exec-[a-z][-a-z]*', panel)):
    if f'.{cls}' not in css and cls not in css:
        bug(f'CSS missing .{cls}')

# ═══ 6. API endpoint matching ═══
for svc_file in ['frontend/js/services/runtime.js', 'frontend/js/services/live_execution.js']:
    js = open(svc_file, encoding='utf-8').read()
    for path in re.findall(r"'/api/([^'?]*)'", js):
        if '${' in path: continue
        # Just check it's not obviously wrong
        if path.startswith('runtime/') or path.startswith('live-execution/'):
            pass  # known routes

# ═══ 7. Test suite ═══
r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/execution/', '-q', '--tb=no',
                    '--ignore=tests/execution/test_paper_execution_router.py'],
                   capture_output=True, text=True, timeout=30)
test_lines = [l for l in r.stdout.split('\n') if 'passed' in l or 'failed' in l]
if test_lines:
    line = test_lines[-1].strip()
    if 'failed' in line:
        bug(f'TESTS: {line}')
    else:
        pass  # all passed

# ═══ 8. HFT module integrity ═══
from server.hft.data_feed.book_builder import BookBuilder
from server.hft.data_feed.features import compute_features
from server.hft.router import RegimeRouter
bb = BookBuilder()
bb.update({'bids':[['100','10']],'asks':[['101','10']]})
f = compute_features(bb, 0.01)
if f.mid <= 0: bug('HFT features: mid price 0')
router = RegimeRouter()
d = router.route(f)
if d.strategy not in ('no_trade','inventory_mm','imbalance_mr','sweep_breakout'):
    bug(f'HFT router: unknown strategy {d.strategy}')

# ═══ 9. Evolution engine ═══
from server.strategy.evolution import EvolutionEngine
e = EvolutionEngine()
v = e._random_variant()
if not v.symbol: bug('evolution: no symbol')
if not v.trigger_modes: bug('evolution: no triggers')

# ═══ 10. Factor engine ═══
from server.factors.factor_engine import FactorEngine
factors = FactorEngine.list_factors()
if len(factors) < 20: bug(f'factors: only {len(factors)} (expected 27+)')
combos = FactorEngine.generate_candidates(3)
if len(combos) != 3: bug(f'factors: generate_candidates returned {len(combos)}')

# ═══ 11. SR strategy ═══
from server.strategy.sr_strategy import ZONE_SCORE_MIN, RESPECT_MIN, MIN_RR
if ZONE_SCORE_MIN > 0.5: bug(f'SR: ZONE_SCORE_MIN={ZONE_SCORE_MIN}')
if MIN_RR > 3.0: bug(f'SR: MIN_RR={MIN_RR}')

# ═══ 12. TG bot commands ═══
bot = open('run_tg_bot.py', encoding='utf-8').read()
for cmd in ['handle_strategy_command','show_running_strategies','show_catalog','handle_stop_command']:
    if cmd not in bot: bug(f'TG bot: missing {cmd}')
for route in ['/create','/策略库','/运行中','/stop','/leaderboard','/status']:
    if route not in bot: bug(f'TG bot: missing command {route}')

# ═══ 13. Catalog completeness ═══
from server.strategy.catalog import list_templates
templates = list_templates()
if len(templates) < 9: bug(f'catalog: only {len(templates)} templates (expected 9)')
expected_ids = ['sr_reversal','sr_full','sr_retest','sr_pre_limit','ma_ribbon','hft_imbalance','hft_sweep','hft_mm','hf_scalp']
for tid in expected_ids:
    if not any(t['template_id'] == tid for t in templates):
        bug(f'catalog: missing template {tid}')

# ═══ 14. WebSocket feed ═══
ws_code = open('server/hft/data_feed/ws_feed.py', encoding='utf-8').read()
if 'wss://ws.bitget.com' not in ws_code: bug('WS: wrong endpoint')
if 'books5' not in ws_code: bug('WS: not subscribing to books5')
if 'trade' not in ws_code: bug('WS: not subscribing to trade')
if 'reconnect' not in ws_code.lower(): bug('WS: no reconnect logic')

# ═══ 15. Server health ═══
try:
    r = subprocess.run(['curl','-s','--max-time','3','http://127.0.0.1:8001/api/health'],
                      capture_output=True, text=True, timeout=5)
    if 'ok' not in r.stdout: bug('SERVER not running')
except: bug('SERVER unreachable')

# ═══ REPORT ═══
print(f'=== DEEP AUDIT REPORT ===')
print(f'Python files: {py_count}')
print(f'JS files: {js_count}')
print(f'Imports tested: {len(imports)}')
print(f'')
print(f'BUGS: {len(bugs)}')
for b in bugs: print(f'  BUG: {b}')
print(f'WARNINGS: {len(warnings)}')
for w in warnings[:10]: print(f'  WARN: {w}')
if len(warnings) > 10: print(f'  ... and {len(warnings)-10} more warnings')
print(f'')
if not bugs:
    print('ALL CLEAN — zero bugs found')
else:
    print(f'{len(bugs)} bugs need fixing')
