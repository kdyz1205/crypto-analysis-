// frontend/js/workbench/patterns.js — draw S/R lines + consolidation zones

const lineRefs = [];

export function clearPatternLines(chart) {
  for (const s of lineRefs) {
    try { chart.removeSeries(s); } catch {}
  }
  lineRefs.length = 0;
}

/**
 * Render support/resistance trendlines on the chart.
 * @param {*} chart  - lightweight-charts instance
 * @param {Array} supportLines - from /api/patterns supportLines
 * @param {Array} resistLines  - from /api/patterns resistanceLines
 * @param {number} maxLines    - 0 = show all
 */
export function drawPatterns(chart, supportLines = [], resistLines = [], maxLines = 0) {
  if (!chart) return;
  clearPatternLines(chart);

  const toDraw = [];
  for (const l of (supportLines || [])) toDraw.push({ ...l, _type: 'support' });
  for (const l of (resistLines || [])) toDraw.push({ ...l, _type: 'resist' });

  const limited = maxLines > 0 ? toDraw.slice(0, maxLines) : toDraw;

  for (const line of limited) {
    try {
      const series = chart.addLineSeries({
        color: line._type === 'support' ? 'rgba(0,230,118,0.8)' : 'rgba(255,23,68,0.8)',
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
      });

      // Line may be specified as two endpoints or list of points
      if (line.t1 != null && line.t2 != null) {
        series.setData([
          { time: Math.floor(line.t1), value: Number(line.v1) },
          { time: Math.floor(line.t2), value: Number(line.v2) },
        ]);
      } else if (Array.isArray(line.points)) {
        series.setData(line.points.map((p) => ({
          time: Math.floor(p.time ?? p[0]),
          value: Number(p.value ?? p[1]),
        })));
      }
      lineRefs.push(series);
    } catch (err) {
      console.warn('[patterns] failed to draw line:', err);
    }
  }
}

/**
 * Render consolidation zones as horizontal price-band series.
 */
export function drawZones(chart, zones = []) {
  if (!chart || !zones?.length) return;
  for (const zone of zones) {
    try {
      const s = chart.addLineSeries({
        color: 'rgba(255,235,59,0.2)',
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      if (zone.t1 != null && zone.t2 != null) {
        s.setData([
          { time: Math.floor(zone.t1), value: Number(zone.mid ?? (zone.top + zone.bottom) / 2) },
          { time: Math.floor(zone.t2), value: Number(zone.mid ?? (zone.top + zone.bottom) / 2) },
        ]);
      }
      lineRefs.push(s);
    } catch (err) {
      console.warn('[patterns] zone draw failed', err);
    }
  }
}

/**
 * Draw horizontal S/R zones as line series (not price lines).
 * Using addLineSeries with autoscaleInfoProvider: null so they
 * don't stretch the Y-axis on zoom.
 */
const zoneSeries = [];

export function clearHorizontalSRZones(chart) {
  for (const s of zoneSeries) {
    try { chart.removeSeries(s); } catch {}
  }
  zoneSeries.length = 0;
}

export function drawHorizontalSRZones(chart, candleSeries, zones = []) {
  if (!chart || !zones?.length) return;
  clearHorizontalSRZones(chart);

  // Get time range from candle data for horizontal line endpoints
  const lastCandle = candleSeries?.data?.()?.at?.(-1);
  const firstCandle = candleSeries?.data?.()?.at?.(0);
  if (!lastCandle || !firstCandle) return;

  const t1 = firstCandle.time;
  const t2 = lastCandle.time;

  for (const zone of zones) {
    try {
      const isSupport = zone.side === 'support';
      const color = isSupport ? 'rgba(0, 230, 118, 0.6)' : 'rgba(255, 23, 68, 0.6)';

      // Center line
      const s = chart.addLineSeries({
        color,
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
        priceLineVisible: false,
        lastValueVisible: true,
        title: `${isSupport ? 'S' : 'R'} ${zone.touches}t`,
        autoscaleInfoProvider: () => null,
      });
      s.setData([
        { time: t1, value: zone.price_center },
        { time: t2, value: zone.price_center },
      ]);
      zoneSeries.push(s);

      // Upper boundary (dotted)
      const sUp = chart.addLineSeries({
        color: color.replace('0.6', '0.25'),
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        priceLineVisible: false,
        lastValueVisible: false,
        autoscaleInfoProvider: () => null,
      });
      sUp.setData([
        { time: t1, value: zone.price_high },
        { time: t2, value: zone.price_high },
      ]);
      zoneSeries.push(sUp);

      // Lower boundary (dotted)
      const sLo = chart.addLineSeries({
        color: color.replace('0.6', '0.25'),
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        priceLineVisible: false,
        lastValueVisible: false,
        autoscaleInfoProvider: () => null,
      });
      sLo.setData([
        { time: t1, value: zone.price_low },
        { time: t2, value: zone.price_low },
      ]);
      zoneSeries.push(sLo);
    } catch (err) {
      console.warn('[patterns] S/R zone draw failed:', err);
    }
  }
}
