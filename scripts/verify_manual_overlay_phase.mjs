import assert from 'node:assert/strict';

import { resolveManualTrendlinePoints } from '../frontend/js/workbench/overlays/manual_trendline_overlay.js';

const baseLine = {
  manual_line_id: 'manual-verify',
  t_start: 200,
  t_end: 300,
  price_start: 100,
  price_end: 110,
  extend_left: false,
  extend_right: false,
};

const leftExtended = resolveManualTrendlinePoints(
  { ...baseLine, extend_left: true },
  { earliestTime: 100, latestTime: 350 },
);
assert.equal(leftExtended[0].time, 100, 'extend_left should project line start to earliest visible time');
assert.equal(leftExtended[0].value, 90, 'extend_left should project price backward');
assert.equal(leftExtended[1].time, 300, 'extend_left should preserve original end when extend_right is off');

const rightExtended = resolveManualTrendlinePoints(
  { ...baseLine, extend_right: true },
  { earliestTime: 100, latestTime: 350 },
);
assert.equal(rightExtended[0].time, 200, 'extend_right should preserve original start when extend_left is off');
assert.equal(rightExtended[1].time, 350, 'extend_right should project line end to latest visible time');
assert.equal(rightExtended[1].value, 115, 'extend_right should project price forward');

const bothExtended = resolveManualTrendlinePoints(
  { ...baseLine, extend_left: true, extend_right: true },
  { earliestTime: 100, latestTime: 350 },
);
assert.deepEqual(
  bothExtended,
  [
    { time: 100, value: 90 },
    { time: 350, value: 115 },
  ],
  'extend_left + extend_right should project to both visible bounds',
);

console.log(JSON.stringify({ passed: true, details: { leftExtended, rightExtended, bothExtended } }, null, 2));
