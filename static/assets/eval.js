/* RAGMill eval page — (a) render the stored measured numbers from
   GET /api/eval/results as stat cards, (b) run a live mini-eval via
   POST /api/eval {limit: 60} (the full golden set; ~10-20 s locally). */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = s => window.RAGMILL.esc(s);
  const nfmt = n => window.RAGMILL.nfmt(n);

  const MODES = [
    ['dense', 'Dense only', 'vector search alone (bge-small embeddings)'],
    ['sparse', 'Sparse only', 'BM25 keyword search alone'],
    ['hybrid', 'Hybrid (RRF)', 'both engines fused with reciprocal rank fusion']
  ];

  function metric(name, v) {
    return '<div class="metric"><span class="mv">' + (v == null ? '—' : v) +
           '</span><span class="ml">' + name + '</span></div>';
  }

  function modeCard(key, label, sub, m, best) {
    return '<div class="mode-card' + (best ? ' best' : '') + '">' +
      '<h3>' + esc(label) +
        (best ? '<span class="tag">best overall</span>' : '') + '</h3>' +
      '<p class="sub">' + esc(sub) + '</p>' +
      '<div class="metrics">' +
        metric('hit@3', m['hit@3']) +
        metric('hit@10', m['hit@10']) +
        metric('MRR', m['mrr']) +
      '</div></div>';
  }

  async function loadStored() {
    try {
      const r = await fetch('/api/eval/results');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const b = await r.json();
      $('storedCards').innerHTML = MODES.map(function (m) {
        return modeCard(m[0], m[1], m[2], b.metrics[m[0]], m[0] === 'hybrid');
      }).join('');
      $('storedMeta').innerHTML =
        'Stored run — ' + nfmt(b.n_queries) + ' golden queries on the 5,000-doc ' +
        'corpus · embedder ' + esc(b.embedder) + ' · ~' +
        b.metrics.latency_ms_per_query + ' ms/query · ' +
        '<a href="/api/eval/results" target="_blank" rel="noopener">raw JSON</a>';
    } catch (e) {
      $('storedCards').innerHTML =
        '<div class="error-box"><b>Stored results unavailable</b>' +
        esc(String(e.message || e)) + '</div>';
    }
  }

  async function runLive() {
    const btn = $('runEvalBtn');
    btn.disabled = true;
    btn.textContent = 'running… replaying 60 golden queries (~10–20 s)';
    $('liveOut').innerHTML =
      '<p class="muted">Replaying the golden set against the live retriever — ' +
      'this page will update when the run finishes.</p>';
    try {
      const r = await fetch('/api/eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 60 })
      });
      if (!r.ok) {
        let msg = 'HTTP ' + r.status;
        try { const b = await r.json(); if (b.detail) msg = b.detail; } catch (e) {}
        throw new Error(msg);
      }
      const b = await r.json();
      const rows = MODES.map(m =>
        '<tr><th>' + esc(m[1]) + '</th>' +
        '<td>' + b.metrics[m[0]]['hit@3'] + '</td>' +
        '<td>' + b.metrics[m[0]]['hit@10'] + '</td>' +
        '<td>' + b.metrics[m[0]]['mrr'] + '</td></tr>').join('');
      $('liveOut').innerHTML =
        '<p class="live-tag">Live run — executed on this server just now · ' +
        nfmt(b.n_queries) + ' queries in ' + b.took_s + ' s</p>' +
        '<table class="eval-table"><thead><tr><th>mode</th><th>hit@3</th>' +
        '<th>hit@10</th><th>MRR</th></tr></thead><tbody>' + rows + '</tbody></table>' +
        '<p class="muted" style="font-size:13px;margin-top:8px">Numbers above are ' +
        'from this request, not the stored file — they reflect the corpus this ' +
        'server has loaded right now.</p>';
    } catch (e) {
      $('liveOut').innerHTML =
        '<div class="error-box"><b>Live eval failed</b>' +
        esc(String(e.message || e)) + '</div>';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Run a live mini-eval';
    }
  }

  $('runEvalBtn').addEventListener('click', runLive);
  loadStored();
})();
