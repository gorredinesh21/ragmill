/* RAGMill shared helpers — plain JS, no dependencies. Loaded on every page. */
(function () {
  'use strict';

  function esc(s) {
    var d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  function nfmt(n) { return Number(n).toLocaleString('en-US'); }

  /* Corpus status pill: "5,000 documents loaded · ..." from /api/healthz. */
  function renderCorpusStatus(el, b) {
    if (b.corpus_docs > 0) {
      el.innerHTML = '<span class="dot on"></span>' + esc(nfmt(b.corpus_docs)) +
        ' documents loaded · embedder ' + esc(b.embedder) +
        ' · answers by ' + esc(b.llm);
    } else {
      el.innerHTML = '<span class="dot wait"></span>corpus is still loading — ' +
        'try again in a few seconds';
    }
  }

  async function loadCorpusStatus(el) {
    if (!el) return;
    try {
      var r = await fetch('/api/healthz');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      renderCorpusStatus(el, await r.json());
    } catch (e) {
      el.innerHTML = '<span class="dot wait"></span>corpus status unavailable (' +
        esc(String(e.message || e)) + ')';
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    loadCorpusStatus(document.getElementById('corpusStatus'));
  });

  window.RAGMILL = {
    esc: esc,
    nfmt: nfmt,
    loadCorpusStatus: loadCorpusStatus
  };
})();
