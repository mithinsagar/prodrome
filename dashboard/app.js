/* ---------------------------------------------------------------------------
   prodrome dashboard.

   Charts are hand-built SVG rather than a charting library, for three reasons:
   the page must be self-contained and free to host, the forms needed here are
   simple (bars, step lines, a scatter), and a library's defaults -- thick marks,
   heavy gridlines, a colour cycle -- are exactly what the visual design rejects.

   Conventions held throughout, from the project's data-viz rules:
     * colour follows the entity, never its rank, so filtering never repaints;
     * every multi-series chart carries a legend AND direct end labels, because
       the palette's worst tritan separation sits in the band where colour alone
       is not sufficient;
     * marks are thin, gridlines are solid hairlines one shade off the surface;
     * values are direct-labelled selectively -- endpoints and extremes only.
--------------------------------------------------------------------------- */

'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';

/* Fixed criterion identity. Assigned by key, never by position in a filtered
   list, so a criterion keeps its colour no matter what else is on screen. */
const CRITERION_STYLE = {
  ema_ror025:       { slot: 'var(--series-1)', label: 'EMA ROR lower bound',   short: 'EMA ROR₀₂₅' },
  who_oe025:        { slot: 'var(--series-3)', label: 'Shrinkage O/E lower bound', short: 'Shrinkage O/E' },
  mhra_prr:         { slot: 'var(--series-2)', label: 'MHRA triple',            short: 'MHRA triple' },
  dubious_prr_only: { slot: 'var(--series-4)', label: 'PRR ≥ 2, no gate (control)', short: 'PRR ≥ 2 only' },
};
const CRITERION_ORDER = ['ema_ror025', 'who_oe025', 'mhra_prr', 'dubious_prr_only'];

let DATA = null;
const tooltip = document.getElementById('tooltip');

/* --------------------------------------------------------------- helpers -- */

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function html(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function svgRoot(container, width, height) {
  container.textContent = '';
  const svg = el('svg', {
    class: 'chart',
    viewBox: `0 0 ${width} ${height}`,
    // A fixed viewBox with preserveAspectRatio scales the whole chart, including
    // the axis band -- which is what keeps axis labels from being cropped by a
    // container sized only to the plot.
    preserveAspectRatio: 'xMinYMin meet',
    role: 'img',
  });
  svg.style.height = 'auto';
  container.appendChild(svg);
  return svg;
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (!Number.isFinite(value)) return '∞';
  return Number(value).toFixed(digits);
}

function fmtInt(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toLocaleString('en-GB');
}

function fmtPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function showTooltip(event, title, rows) {
  tooltip.textContent = '';
  tooltip.appendChild(html('div', 'tt-title', title));
  for (const [key, value] of rows) {
    const row = html('div', 'tt-row');
    row.appendChild(html('span', null, key));
    row.appendChild(html('span', null, value));
    tooltip.appendChild(row);
  }
  tooltip.classList.add('visible');
  moveTooltip(event);
}

function moveTooltip(event) {
  // Flip near the right/bottom edge so the tooltip never leaves the viewport.
  const pad = 14;
  const box = tooltip.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - pad;
  if (y + box.height > window.innerHeight - 8) y = event.clientY - box.height - pad;
  tooltip.style.left = `${Math.max(8, x)}px`;
  tooltip.style.top = `${Math.max(8, y)}px`;
}

function hideTooltip() { tooltip.classList.remove('visible'); }

function hoverable(node, title, rows) {
  node.addEventListener('mouseenter', (e) => showTooltip(e, title, rows));
  node.addEventListener('mousemove', moveTooltip);
  node.addEventListener('mouseleave', hideTooltip);
}

function legendInto(container, items) {
  container.textContent = '';
  for (const item of items) {
    const wrap = html('span', 'legend-item');
    const key = html('span', 'legend-key');
    key.style.background = item.colour;
    if (item.dashed) { key.style.background = 'transparent'; key.style.borderTop = `2px dashed ${item.colour}`; }
    wrap.appendChild(key);
    wrap.appendChild(html('span', null, item.label));
    container.appendChild(wrap);
  }
}

function emptyInto(container, message) {
  container.textContent = '';
  container.appendChild(html('p', 'empty', message));
}

function quarterIndex(label) {
  const match = /^(\d{4})Q([1-4])$/.exec(label || '');
  if (!match) return null;
  return Number(match[1]) * 4 + Number(match[2]) - 1;
}

/* ------------------------------------------------------------- manifest -- */

function renderManifest() {
  const m = DATA.manifest || {};
  const target = document.getElementById('manifest');
  target.textContent = '';
  const entries = [
    ['run', m.run_id ? String(m.run_id).slice(0, 10) : '—'],
    ['window', m.first_quarter && m.last_quarter ? `${m.first_quarter}–${m.last_quarter}` : '—'],
    ['openFDA data to', m.openfda_last_updated || '—'],
    ['drugs', m.drugs != null ? fmtInt(m.drugs) : '—'],
    ['point-in-time cells', m.contingency_cells != null ? fmtInt(m.contingency_cells) : '—'],
    ['embeddings', m.embed_backend || '—'],
    ['requests', m.requests != null ? fmtInt(m.requests) : '—'],
    ['version', m.prodrome_version || '—'],
  ];
  for (const [key, value] of entries) {
    const span = html('span');
    span.appendChild(document.createTextNode(`${key} `));
    const strong = html('strong', null, value);
    span.appendChild(strong);
    target.appendChild(span);
  }
}

/* --------------------------------------------------------------- thesis -- */

function renderThesis() {
  const leakage = DATA.leakage || {};
  const m = DATA.manifest || {};
  const criteria = DATA.criteria || [];

  const inflation = leakage.median_inflation;
  const heroNode = document.getElementById('hero-inflation');
  heroNode.textContent = '';
  heroNode.appendChild(document.createTextNode(inflation != null ? fmt(inflation, 2) : '—'));
  heroNode.appendChild(html('span', 'hero-unit', '×'));

  document.getElementById('tile-gaps').textContent = fmtInt((DATA.gaps || []).length
    ? (m.total_open_gaps != null ? m.total_open_gaps : DATA.gaps.length)
    : 0);

  const best = criteria
    .filter((c) => c.median_lead_quarters != null)
    .sort((a, b) => b.median_lead_quarters - a.median_lead_quarters)[0];
  document.getElementById('tile-lead').textContent = best ? fmt(best.median_lead_quarters, 1) : '—';
  document.getElementById('tile-lead-sub').textContent = best
    ? `quarters — ${(CRITERION_STYLE[best.criterion] || {}).short || best.criterion}`
    : 'quarters';

  document.getElementById('tile-notorious').textContent = fmtInt(leakage.n_notorious);
  document.getElementById('tile-cohort').textContent = fmtInt(m.drugs);
  document.getElementById('tile-cohort-sub').textContent =
    m.drugs_reliant_on_name_matching != null
      ? `${m.drugs_reliant_on_name_matching} rely on name matching`
      : '';
  document.getElementById('tile-lift').textContent =
    m.model_lift_at_50 != null ? `${fmt(m.model_lift_at_50, 1)}×` : '—';
  document.getElementById('tile-lift-sub').textContent =
    m.model_precision_at_50 != null && m.model_base_rate != null
      ? `${fmtPct(m.model_precision_at_50)} vs ${fmtPct(m.model_base_rate, 2)}`
      : 'vs. base rate, held out';
  document.getElementById('tile-quarters').textContent = fmtInt(m.quarters);

  const caveat = document.getElementById('leakage-caveat');
  if (leakage.n_pairs) {
    caveat.textContent =
      `Across ${fmtInt(leakage.n_pairs)} pairs that were eventually labelled, ` +
      `${fmtInt(leakage.n_materially_inflated)} show a retrospective statistic at least ` +
      `twice the value available at the label change, and ${fmtInt(leakage.n_notorious)} ` +
      `show reporting more than doubling in the year after it. A retrospective analysis ` +
      `would present those inflated figures as evidence the signal was detectable.`;
  } else {
    caveat.textContent = '';
  }
}

/* ---------------------------------------------- criterion lead-time bars -- */

function renderLeadTimeChart() {
  const container = document.getElementById('chart-leadtime');
  const rows = CRITERION_ORDER
    .map((key) => (DATA.criteria || []).find((c) => c.criterion === key))
    .filter(Boolean);
  if (!rows.length) { emptyInto(container, 'No criterion results in this run.'); return; }

  const W = 560, rowH = 46, padL = 132, padR = 92, padT = 8, padB = 30;
  const H = padT + rows.length * rowH + padB;
  const svg = svgRoot(container, W, H);
  const plotW = W - padL - padR;
  const maxValue = Math.max(1, ...rows.map((r) => r.median_lead_quarters || 0)) * 1.12;
  const x = (v) => padL + (v / maxValue) * plotW;

  // Hairline grid, solid -- never dashed.
  for (let t = 0; t <= maxValue; t += maxValue > 8 ? 4 : 2) {
    svg.appendChild(el('line', { class: 'grid-line', x1: x(t), x2: x(t), y1: padT, y2: padT + rows.length * rowH }));
    svg.appendChild(el('text', { x: x(t), y: H - padB + 16, 'text-anchor': 'middle' }, String(t)));
  }
  svg.appendChild(el('text', { x: padL + plotW / 2, y: H - 2, 'text-anchor': 'middle', class: 'axis-label' }, 'quarters to label change'));

  rows.forEach((row, index) => {
    const style = CRITERION_STYLE[row.criterion] || { slot: 'var(--status-neutral)', short: row.criterion };
    const yCentre = padT + index * rowH + rowH / 2;
    const value = row.median_lead_quarters;

    svg.appendChild(el('text', {
      x: padL - 10, y: yCentre + 4, 'text-anchor': 'end', class: 'series-label', fill: 'var(--ink-2)',
    }, style.short));

    if (value == null) {
      svg.appendChild(el('text', { x: padL + 6, y: yCentre + 4, class: 'value-label', fill: 'var(--ink-4)' },
        'never reached the median'));
      return;
    }

    // 4px rounded data-end anchored to the baseline: rounded on the value end only.
    const barH = 14;
    const bar = el('path', {
      d: roundedRightBar(padL, yCentre - barH / 2, x(value) - padL, barH, 4),
      fill: style.slot,
    });
    hoverable(bar, style.label, [
      ['median lead time', `${fmt(value, 1)} quarters`],
      ['≈ months', fmt(row.median_lead_months, 0)],
      ['signals later labelled', fmtInt(row.signalled_then_labelled)],
      ['still open', fmtInt(row.signalled_not_yet_labelled)],
      ['labelled before signalling', fmtInt(row.labelled_before_signal)],
      ['share of signals labelled', fmtPct(row.share_of_signals_labelled)],
    ]);
    svg.appendChild(bar);

    // Direct label at the data end -- selective, not on every mark.
    svg.appendChild(el('text', {
      x: x(value) + 8, y: yCentre + 4, class: 'value-label', fill: 'var(--ink-1)',
    }, `${fmt(value, 1)} q`));
    svg.appendChild(el('text', {
      x: x(value) + 8, y: yCentre + 16, class: 'value-label', fill: 'var(--ink-4)', 'font-size': 9,
    }, `${fmtInt(row.signalled_not_yet_labelled)} open`));
  });
}

function roundedRightBar(x, y, width, height, radius) {
  const w = Math.max(width, radius + 0.5);
  const r = Math.min(radius, height / 2, w);
  return `M ${x} ${y} H ${x + w - r} Q ${x + w} ${y} ${x + w} ${y + r} V ${y + height - r} Q ${x + w} ${y + height} ${x + w - r} ${y + height} H ${x} Z`;
}

/* ------------------------------------------------------- survival curves -- */

function renderSurvivalChart() {
  const container = document.getElementById('chart-survival');
  const points = DATA.survival || [];
  if (!points.length) { emptyInto(container, 'No survival curves in this run.'); return; }

  const byCriterion = new Map();
  for (const point of points) {
    if (!byCriterion.has(point.criterion)) byCriterion.set(point.criterion, []);
    byCriterion.get(point.criterion).push(point);
  }
  const series = CRITERION_ORDER
    .filter((key) => byCriterion.has(key))
    .map((key) => ({
      key,
      style: CRITERION_STYLE[key],
      points: byCriterion.get(key).slice().sort((a, b) => a.quarters - b.quarters),
    }));
  if (!series.length) { emptyInto(container, 'No survival curves in this run.'); return; }

  legendInto(document.getElementById('legend-survival'),
    series.map((s) => ({ colour: s.style.slot, label: s.style.short })));

  const W = 560, H = 300, padL = 44, padR = 86, padT = 10, padB = 42;
  const svg = svgRoot(container, W, H);
  const maxQ = Math.max(4, ...series.flatMap((s) => s.points.map((p) => p.quarters)));
  const x = (q) => padL + (q / maxQ) * (W - padL - padR);
  const y = (p) => padT + (1 - p) * (H - padT - padB);

  for (let p = 0; p <= 1.0001; p += 0.25) {
    svg.appendChild(el('line', { class: 'grid-line', x1: padL, x2: W - padR, y1: y(p), y2: y(p) }));
    svg.appendChild(el('text', { x: padL - 8, y: y(p) + 3, 'text-anchor': 'end' }, `${Math.round(p * 100)}%`));
  }
  const step = maxQ > 20 ? 8 : 4;
  for (let q = 0; q <= maxQ; q += step) {
    svg.appendChild(el('text', { x: x(q), y: H - padB + 16, 'text-anchor': 'middle' }, String(q)));
  }
  svg.appendChild(el('line', { class: 'axis-line', x1: padL, x2: W - padR, y1: y(0), y2: y(0) }));
  svg.appendChild(el('text', { x: padL + (W - padL - padR) / 2, y: H - 6, 'text-anchor': 'middle', class: 'axis-label' },
    'quarters since the criterion first fired'));

  for (const s of series) {
    // A survival function is a step function; drawing it as a smooth line would
    // imply events between observation times that were not observed.
    let path = '';
    let previous = 1;
    for (const point of s.points) {
      const incidence = 1 - point.survival;
      path += path === ''
        ? `M ${x(0)} ${y(0)} L ${x(point.quarters)} ${y(1 - previous)}`
        : ` L ${x(point.quarters)} ${y(1 - previous)}`;
      path += ` L ${x(point.quarters)} ${y(incidence)}`;
      previous = point.survival;
    }
    svg.appendChild(el('path', { d: path, fill: 'none', stroke: s.style.slot, 'stroke-width': 2, 'stroke-linejoin': 'round' }));

    const last = s.points[s.points.length - 1];
    const endY = y(1 - last.survival);
    // Direct end label, so identity is never carried by colour alone.
    svg.appendChild(el('text', {
      x: x(last.quarters) + 7, y: endY + 3, class: 'series-label', fill: s.style.slot,
    }, s.style.short));

    // Wide invisible hit area: the hover target must be bigger than a 2px line.
    const hit = el('path', { d: path, fill: 'none', stroke: 'transparent', 'stroke-width': 14 });
    const curve = (DATA.criteria || []).find((c) => c.criterion === s.key) || {};
    hoverable(hit, s.style.label, [
      ['labelled within 1 year', fmtPct(curve.labelled_by_1y)],
      ['within 2 years', fmtPct(curve.labelled_by_2y)],
      ['within 3 years', fmtPct(curve.labelled_by_3y)],
      ['pairs at risk', fmtInt(curve.signalled_then_labelled + curve.signalled_not_yet_labelled)],
    ]);
    svg.appendChild(hit);
  }
}

/* ------------------------------------------------------- criteria table -- */

function renderCriteriaTable() {
  const table = document.getElementById('table-criteria');
  const rows = CRITERION_ORDER
    .map((key) => (DATA.criteria || []).find((c) => c.criterion === key))
    .filter(Boolean);
  table.textContent = '';
  if (!rows.length) { table.appendChild(html('caption', 'empty', 'No criterion results.')); return; }

  const columns = [
    ['Criterion', 'text'],
    ['Median lead (q)', 'num'],
    ['Labelled by 2y', 'num'],
    ['Signals → labelled', 'num'],
    ['Still open', 'num'],
    ['Labelled first', 'num'],
    ['Never signalled', 'num'],
    ['Excluded (truncated)', 'num'],
    ['Firing stability', 'num'],
  ];
  const thead = html('thead');
  const headRow = html('tr');
  for (const [label, kind] of columns) headRow.appendChild(html('th', kind === 'num' ? 'num' : null, label));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = html('tbody');
  for (const row of rows) {
    const style = CRITERION_STYLE[row.criterion] || { slot: 'var(--status-neutral)', short: row.criterion };
    const tr = html('tr');
    const nameCell = html('td');
    const swatch = html('span', 'swatch');
    swatch.style.background = style.slot;
    nameCell.appendChild(swatch);
    nameCell.appendChild(document.createTextNode(style.short));
    if (row.criterion === 'dubious_prr_only') {
      nameCell.appendChild(document.createTextNode(' '));
      nameCell.appendChild(html('span', 'chip neutral', 'control'));
    }
    tr.appendChild(nameCell);
    for (const value of [
      row.median_lead_quarters != null ? fmt(row.median_lead_quarters, 1) : '—',
      fmtPct(row.labelled_by_2y),
      fmtInt(row.signalled_then_labelled),
      fmtInt(row.signalled_not_yet_labelled),
      fmtInt(row.labelled_before_signal),
      fmtInt(row.never_signalled),
      fmtInt(row.excluded_left_truncated),
      fmtPct(row.mean_firing_fraction),
    ]) tr.appendChild(html('td', 'num', value));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

/* ---------------------------------------------------------- queue table -- */

function renderQueue() {
  const table = document.getElementById('table-queue');
  const drugFilter = document.getElementById('filter-drug').value;
  const artefactFilter = document.getElementById('filter-artefact').value;
  const minReports = Number(document.getElementById('filter-min-reports').value) || 1;

  let rows = (DATA.gaps || []).filter((g) => (g.reports || 0) >= minReports);
  if (drugFilter) rows = rows.filter((g) => g.drug_unii === drugFilter);
  if (artefactFilter === 'clean') rows = rows.filter((g) => !g.artefact_flags);
  if (artefactFilter === 'flagged') rows = rows.filter((g) => !!g.artefact_flags);

  table.textContent = '';
  const columns = [
    ['#', 'num'], ['Drug', 'text'], ['Reaction', 'text'], ['Reports', 'num'],
    ['Expected', 'num'], ['ROR (95% low)', 'num'], ['EB05', 'num'],
    ['Calibrated p', 'num'], ['First fired', 'text'], ['P(label ≤ 8q)', 'num'],
    ['Reporting pattern', 'text'],
  ];
  const thead = html('thead');
  const headRow = html('tr');
  for (const [label, kind] of columns) headRow.appendChild(html('th', kind === 'num' ? 'num' : null, label));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = html('tbody');
  if (!rows.length) {
    const tr = html('tr');
    const td = html('td', 'empty', 'No open label gaps match these filters.');
    td.setAttribute('colspan', String(columns.length));
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
  rows.slice(0, 250).forEach((gap, index) => {
    const tr = html('tr');
    tr.appendChild(html('td', 'num', String(index + 1)));
    tr.appendChild(html('td', 'drug', gap.drug_name));
    tr.appendChild(html('td', 'reaction', titleCase(gap.reaction)));
    tr.appendChild(html('td', 'num', fmtInt(gap.reports)));
    tr.appendChild(html('td', 'num', fmt(gap.expected, 1)));
    tr.appendChild(html('td', 'num',
      gap.ror != null ? `${fmt(gap.ror, 2)} (${fmt(gap.ror_ci_lower, 2)})` : '—'));
    tr.appendChild(html('td', 'num', fmt(gap.eb05, 2)));

    const pCell = html('td', 'num', gap.calibrated_p != null ? fmt(gap.calibrated_p, 3) : '—');
    if (gap.significance_lost_to_calibration) {
      pCell.appendChild(document.createTextNode(' '));
      pCell.appendChild(html('span', 'chip neutral', 'nominal only'));
    }
    tr.appendChild(pCell);

    tr.appendChild(html('td', null, gap.signal_quarter || '—'));
    tr.appendChild(html('td', 'num',
      gap.hazard_probability != null ? fmtPct(gap.hazard_probability, 1) : '—'));

    const flagCell = html('td');
    const flags = (gap.artefact_flags || '').split(',').map((f) => f.trim()).filter(Boolean);
    if (!flags.length) {
      flagCell.appendChild(html('span', 'chip good', 'clean'));
    } else {
      for (const flag of flags) {
        flagCell.appendChild(html('span', 'chip serious', FLAG_LABEL[flag] || flag));
      }
    }
    if (gap.relies_on_name_matching) {
      flagCell.appendChild(html('span', 'chip neutral', 'name-matched'));
    }
    tr.appendChild(flagCell);

    hoverable(tr, `${gap.drug_name} — ${titleCase(gap.reaction)}`, [
      ['reports', fmtInt(gap.reports)],
      ['expected if independent', fmt(gap.expected, 2)],
      ['PRR', fmt(gap.prr, 2)],
      ['EBGM', fmt(gap.ebgm, 2)],
      ['EB05', fmt(gap.eb05, 2)],
      ['criteria firing', fmtInt(gap.n_criteria_firing)],
      ['robustness', fmt(gap.robustness_score, 2)],
    ]);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

const FLAG_LABEL = {
  geographic_concentration: 'geographic',
  litigation_pattern: 'litigation',
  single_quarter_spike: 'one-quarter spike',
  consumer_dominated: 'consumer-reported',
};

function titleCase(text) {
  return String(text || '').toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ---------------------------------------------------------- trend chart -- */

function renderTrendChart() {
  const container = document.getElementById('chart-trend');
  const select = document.getElementById('select-pair');
  const rows = (DATA.trend || []).filter(
    (r) => `${r.drug_unii}|${r.reaction}` === select.value);
  if (!rows.length) { emptyInto(container, 'Select a pair to see its trajectory.'); return; }

  rows.sort((a, b) => quarterIndex(a.as_of_quarter) - quarterIndex(b.as_of_quarter));

  legendInto(document.getElementById('legend-trend'), [
    { colour: 'var(--series-1)', label: 'ROR (point-in-time)' },
    { colour: 'var(--series-3)', label: '95% lower bound' },
    { colour: 'var(--status-serious)', label: 'label change detected', dashed: true },
  ]);

  const W = 1100, H = 320, padL = 48, padR = 58, padT = 14, padB = 46;
  const svg = svgRoot(container, W, H);
  const plotW = W - padL - padR, plotH = H - padT - padB;

  const values = rows.flatMap((r) => [r.ror, r.ror_ci_lower].filter((v) => v != null && Number.isFinite(v)));
  const maxY = Math.max(2.2, ...values) * 1.1;
  const x = (i) => padL + (rows.length === 1 ? plotW / 2 : (i / (rows.length - 1)) * plotW);
  const y = (v) => padT + plotH - (Math.min(v, maxY) / maxY) * plotH;

  const ticks = 5;
  for (let t = 0; t <= ticks; t += 1) {
    const value = (maxY / ticks) * t;
    svg.appendChild(el('line', { class: 'grid-line', x1: padL, x2: W - padR, y1: y(value), y2: y(value) }));
    svg.appendChild(el('text', { x: padL - 8, y: y(value) + 3, 'text-anchor': 'end' }, fmt(value, 1)));
  }
  // The reference line at ROR = 1 is the "no association" level, so it is drawn
  // as a labelled axis rule rather than as another gridline.
  svg.appendChild(el('line', { class: 'axis-line', x1: padL, x2: W - padR, y1: y(1), y2: y(1) }));
  svg.appendChild(el('text', { x: W - padR + 6, y: y(1) + 3, fill: 'var(--ink-3)' }, 'ROR = 1'));

  const labelEvery = Math.max(1, Math.ceil(rows.length / 14));
  rows.forEach((row, index) => {
    if (index % labelEvery === 0) {
      svg.appendChild(el('text', {
        x: x(index), y: H - padB + 16, 'text-anchor': 'middle',
      }, row.as_of_quarter));
    }
  });
  svg.appendChild(el('text', {
    x: padL + plotW / 2, y: H - 6, 'text-anchor': 'middle', class: 'axis-label',
  }, 'quarter end (cumulative reports received by this date)'));

  const line = (accessor, colour, dash) => {
    const usable = rows.map((r, i) => [i, accessor(r)]).filter(([, v]) => v != null && Number.isFinite(v));
    if (usable.length < 2) return;
    const d = usable.map(([i, v], k) => `${k === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ');
    svg.appendChild(el('path', {
      d, fill: 'none', stroke: colour, 'stroke-width': 2,
      'stroke-dasharray': dash || null, 'stroke-linejoin': 'round',
    }));
  };
  line((r) => r.ror_ci_lower, 'var(--series-3)');
  line((r) => r.ror, 'var(--series-1)');

  // Label-change marker: the event whose distance from first firing is the lead time.
  rows.forEach((row, index) => {
    if (row.transition_type !== 'added') return;
    svg.appendChild(el('line', {
      x1: x(index), x2: x(index), y1: padT, y2: padT + plotH,
      stroke: 'var(--status-serious)', 'stroke-width': 2, 'stroke-dasharray': '4 3',
    }));
    const flag = el('text', {
      x: x(index) + 5, y: padT + 11, fill: 'var(--status-serious)',
      class: 'series-label',
    }, `label change ${row.as_of_quarter}`);
    svg.appendChild(flag);
  });

  // Per-quarter hover: a wide invisible band, so the target is bigger than the mark.
  const bandW = plotW / Math.max(rows.length, 1);
  rows.forEach((row, index) => {
    const band = el('rect', {
      x: x(index) - bandW / 2, y: padT, width: bandW, height: plotH, fill: 'transparent',
    });
    hoverable(band, `${row.as_of_quarter} — cumulative`, [
      ['reports (a)', fmtInt(row.reports)],
      ['expected', fmt(row.expected, 2)],
      ['ROR', fmt(row.ror, 2)],
      ['ROR 95% lower', fmt(row.ror_ci_lower, 2)],
      ['PRR', fmt(row.prr, 2)],
      ['EB05', fmt(row.eb05, 2)],
      ['χ² (Yates)', fmt(row.chi2_yates, 1)],
      ['criteria firing', fmtInt(row.n_criteria_firing)],
      ['on the label then', row.labelled_at_quarter ? 'yes' : 'no'],
    ]);
    svg.appendChild(band);
  });
}

/* ---------------------------------------------------- calibration chart -- */

function renderCalibrationChart() {
  const container = document.getElementById('chart-calibration');
  const rows = (DATA.calibration || []).filter(
    (r) => r.mean_predicted != null && r.observed_rate != null);
  if (!rows.length) { emptyInto(container, 'The model was not fitted in this run.'); return; }

  legendInto(document.getElementById('legend-calibration'), [
    { colour: 'var(--series-1)', label: 'held-out deciles' },
    { colour: 'var(--ink-4)', label: 'perfect calibration', dashed: true },
  ]);

  const W = 520, H = 300, pad = 46;
  const svg = svgRoot(container, W, H);
  const maxV = Math.max(...rows.flatMap((r) => [r.mean_predicted, r.observed_rate])) * 1.15 || 0.1;
  const x = (v) => pad + (v / maxV) * (W - pad * 1.4);
  const y = (v) => H - pad - (v / maxV) * (H - pad * 1.5);

  for (let t = 0; t <= 4; t += 1) {
    const v = (maxV / 4) * t;
    svg.appendChild(el('line', { class: 'grid-line', x1: pad, x2: W - pad * 0.4, y1: y(v), y2: y(v) }));
    svg.appendChild(el('text', { x: pad - 8, y: y(v) + 3, 'text-anchor': 'end' }, fmtPct(v, 1)));
    svg.appendChild(el('text', { x: x(v), y: H - pad + 16, 'text-anchor': 'middle' }, fmtPct(v, 1)));
  }
  svg.appendChild(el('line', {
    x1: x(0), y1: y(0), x2: x(maxV), y2: y(maxV),
    stroke: 'var(--ink-4)', 'stroke-width': 1.5, 'stroke-dasharray': '4 3',
  }));
  svg.appendChild(el('text', { x: pad + (W - pad * 1.4) / 2, y: H - 8, 'text-anchor': 'middle', class: 'axis-label' },
    'predicted probability of a label change'));
  svg.appendChild(el('text', {
    x: 12, y: H / 2, 'text-anchor': 'middle', class: 'axis-label',
    transform: `rotate(-90 12 ${H / 2})`,
  }, 'observed rate'));

  for (const row of rows) {
    const marker = el('circle', {
      cx: x(row.mean_predicted), cy: y(row.observed_rate), r: 5,
      fill: 'var(--series-1)', stroke: 'var(--surface-0)', 'stroke-width': 2,
    });
    hoverable(marker, `Decile ${row.decile + 1}`, [
      ['mean predicted', fmtPct(row.mean_predicted, 2)],
      ['observed', fmtPct(row.observed_rate, 2)],
      ['rows', fmtInt(row.n_rows)],
    ]);
    svg.appendChild(marker);
  }
}

/* --------------------------------------------------- coefficients chart -- */

function renderCoefficientsChart() {
  const container = document.getElementById('chart-coefficients');
  const rows = (DATA.coefficients || [])
    .slice()
    .sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient))
    .slice(0, 10);
  if (!rows.length) { emptyInto(container, 'The model was not fitted in this run.'); return; }

  const W = 520, rowH = 24, padL = 180, padR = 54, padT = 10, padB = 28;
  const H = padT + rows.length * rowH + padB;
  const svg = svgRoot(container, W, H);
  const plotW = W - padL - padR;
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.coefficient))) * 1.1 || 1;
  const zero = padL + plotW / 2;
  const x = (v) => zero + (v / maxAbs) * (plotW / 2);

  svg.appendChild(el('line', { class: 'axis-line', x1: zero, x2: zero, y1: padT, y2: padT + rows.length * rowH }));
  svg.appendChild(el('text', { x: zero, y: H - 4, 'text-anchor': 'middle', class: 'axis-label' },
    'standardised log-odds contribution'));

  rows.forEach((row, index) => {
    const yCentre = padT + index * rowH + rowH / 2;
    // Diverging encoding: sign is polarity, so two hues either side of a neutral
    // zero rule -- not a sequential ramp.
    const positive = row.coefficient >= 0;
    const colour = positive ? 'var(--series-1)' : 'var(--series-4)';
    const left = positive ? zero : x(row.coefficient);
    const width = Math.abs(x(row.coefficient) - zero);
    const bar = el('rect', { x: left, y: yCentre - 6, width: Math.max(width, 1), height: 12, fill: colour, rx: 1 });
    hoverable(bar, row.feature, [
      ['coefficient', fmt(row.coefficient, 3)],
      ['direction', positive ? 'raises the probability' : 'lowers it'],
    ]);
    svg.appendChild(bar);
    svg.appendChild(el('text', { x: padL - 10, y: yCentre + 4, 'text-anchor': 'end', fill: 'var(--ink-2)' },
      FEATURE_LABEL[row.feature] || row.feature));
    svg.appendChild(el('text', {
      x: positive ? x(row.coefficient) + 6 : x(row.coefficient) - 6,
      y: yCentre + 4, 'text-anchor': positive ? 'start' : 'end', class: 'value-label',
    }, fmt(row.coefficient, 2)));
  });
}

const FEATURE_LABEL = {
  log2_oe_shrunk: 'shrinkage log O/E',
  log_ebgm: 'EBGM (log)',
  log_eb05: 'EB05 (log)',
  log_prr: 'PRR (log)',
  log_ror_lower: 'ROR lower bound (log)',
  chi2_yates: 'χ² (Yates)',
  log_a: 'report count (log)',
  log_expected: 'expected count (log)',
  quarters_since_signal: 'quarters since signal',
  firing_fraction: 'firing stability',
  reporter_concentration: 'reporter concentration',
  consumer_share: 'consumer-reported share',
  spike_ratio: 'volume spike ratio',
  n_criteria_firing: 'criteria firing',
  drug_report_share: 'drug share of reports',
};

/* ---------------------------------------------------------- drugs table -- */

function renderDrugsTable() {
  const table = document.getElementById('table-drugs');
  const rows = (DATA.drugs || []).slice();
  table.textContent = '';
  if (!rows.length) { table.appendChild(html('caption', 'empty', 'No cohort in this run.')); return; }

  const columns = [
    ['Drug', 'text'], ['Open gaps', 'num'], ['Signals labelled', 'num'],
    ['Open signals', 'num'], ['Median lead (q)', 'num'], ['Reactions', 'num'],
    ['Reports', 'num'], ['Label versions', 'num'], ['UNII coverage', 'num'],
  ];
  const thead = html('thead');
  const headRow = html('tr');
  for (const [label, kind] of columns) headRow.appendChild(html('th', kind === 'num' ? 'num' : null, label));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = html('tbody');
  for (const row of rows) {
    const tr = html('tr');
    tr.appendChild(html('td', 'drug', row.drug_name));
    tr.appendChild(html('td', 'num', fmtInt(row.open_label_gaps)));
    tr.appendChild(html('td', 'num', fmtInt(row.signals_that_were_labelled)));
    tr.appendChild(html('td', 'num', fmtInt(row.open_signals)));
    tr.appendChild(html('td', 'num',
      row.median_lead_quarters != null ? fmt(row.median_lead_quarters, 1) : '—'));
    tr.appendChild(html('td', 'num', fmtInt(row.reactions_tracked)));
    tr.appendChild(html('td', 'num', fmtInt(row.latest_drug_reports)));
    tr.appendChild(html('td', 'num', fmtInt(row.usable_label_versions)));

    const coverage = html('td', 'num',
      row.unii_coverage != null ? fmtPct(row.unii_coverage, 0) : '—');
    if (row.relies_on_name_matching) {
      coverage.appendChild(document.createTextNode(' '));
      coverage.appendChild(html('span', 'chip neutral', 'name'));
    }
    tr.appendChild(coverage);
    hoverable(tr, row.drug_name, [
      ['first archived label', row.first_label_date || '—'],
      ['latest label', row.latest_label_date || '—'],
      ['already labelled at baseline', fmtInt(row.already_labelled_at_baseline)],
      ['notes', row.notes || '—'],
    ]);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

/* --------------------------------------------------------------- wiring -- */

function populateFilters() {
  const drugSelect = document.getElementById('filter-drug');
  const seen = new Map();
  for (const gap of DATA.gaps || []) {
    if (!seen.has(gap.drug_unii)) seen.set(gap.drug_unii, gap.drug_name);
  }
  for (const [unii, name] of [...seen].sort((a, b) => a[1].localeCompare(b[1]))) {
    const option = document.createElement('option');
    option.value = unii;
    option.textContent = name;
    drugSelect.appendChild(option);
  }

  const pairSelect = document.getElementById('select-pair');
  const pairs = new Map();
  for (const row of DATA.trend || []) {
    const key = `${row.drug_unii}|${row.reaction}`;
    if (!pairs.has(key)) pairs.set(key, `${row.drug_name} — ${titleCase(row.reaction)}`);
  }
  for (const [key, label] of [...pairs].sort((a, b) => a[1].localeCompare(b[1]))) {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = label;
    pairSelect.appendChild(option);
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('prodrome-theme', theme); } catch { /* private mode */ }
  // Charts read CSS custom properties at build time, so they must be redrawn.
  if (DATA) renderCharts();
}

function renderCharts() {
  renderLeadTimeChart();
  renderSurvivalChart();
  renderTrendChart();
  renderCalibrationChart();
  renderCoefficientsChart();
}

function renderAll() {
  renderManifest();
  renderThesis();
  renderCriteriaTable();
  renderQueue();
  renderDrugsTable();
  renderCharts();
}

async function main() {
  let stored = null;
  try { stored = localStorage.getItem('prodrome-theme'); } catch { /* private mode */ }
  document.documentElement.setAttribute('data-theme', stored || 'dark');

  document.getElementById('theme-toggle').addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });

  try {
    const response = await fetch('data/dashboard.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    DATA = await response.json();
  } catch (error) {
    document.getElementById('manifest').textContent =
      `Could not load data/dashboard.json (${error.message}). Run: prodrome export`;
    return;
  }

  populateFilters();
  renderAll();

  for (const id of ['filter-drug', 'filter-artefact', 'filter-min-reports']) {
    document.getElementById(id).addEventListener('input', renderQueue);
  }
  document.getElementById('select-pair').addEventListener('change', renderTrendChart);
  window.addEventListener('resize', () => { /* viewBox scales; nothing to redo */ });
}

main();
