/* RAGMill sources page — browse/search the loaded corpus via GET /api/documents.
   Plain substring search (server-side), paginated; cards expand to full text. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = s => window.RAGMILL.esc(s);
  const nfmt = n => window.RAGMILL.nfmt(n);

  let q = '';
  let page = 1;
  let pages = 1;

  async function load() {
    $('srcMeta').textContent = 'loading…';
    const params = new URLSearchParams();
    if (q) params.set('q', q);
    params.set('page', String(page));
    try {
      const r = await fetch('/api/documents?' + params.toString());
      if (!r.ok) {
        let msg = 'HTTP ' + r.status;
        try { const b = await r.json(); if (b.detail) msg = b.detail; } catch (e) {}
        throw new Error(msg);
      }
      render(await r.json());
    } catch (e) {
      $('srcMeta').textContent = '';
      $('docList').innerHTML =
        '<div class="error-box"><b>Could not load documents</b>' +
        esc(String(e.message || e)) + '</div>';
      $('pageInfo').textContent = '';
      $('prevBtn').disabled = true;
      $('nextBtn').disabled = true;
    }
  }

  function render(b) {
    pages = b.pages;
    const from = b.total === 0 ? 0 : (b.page - 1) * b.per_page + 1;
    const to = Math.min(b.page * b.per_page, b.total);
    $('srcMeta').innerHTML = b.q
      ? '<b>' + nfmt(b.total) + '</b> documents match “' + esc(b.q) +
        '” · showing ' + from + '–' + to
      : '<b>' + nfmt(b.total) + '</b> documents in the corpus · showing ' +
        from + '–' + to;
    $('docList').innerHTML = b.documents.length
      ? b.documents.map(d =>
          '<details class="doc">' +
            '<summary>' +
              '<span class="doc-title">' + esc(d.title) + '</span>' +
              '<span class="doc-meta">' + esc(d.arxiv_id) + ' · ' +
                esc((d.categories || []).join(' ')) + '</span>' +
              '<span class="doc-snip">' + esc(d.snippet) + '</span>' +
            '</summary>' +
            '<div class="doc-body"><p>' + esc(d.abstract) + '</p>' +
              '<a href="' + esc(d.link) + '" target="_blank" rel="noopener">' +
              'view source ↗</a></div>' +
          '</details>').join('')
      : '<p class="muted">No documents match that filter.</p>';
    $('pageInfo').textContent = 'page ' + b.page + ' of ' + b.pages;
    $('prevBtn').disabled = b.page <= 1;
    $('nextBtn').disabled = b.page >= b.pages;
  }

  $('srcForm').addEventListener('submit', e => {
    e.preventDefault();
    q = $('srcQ').value.trim();
    page = 1;
    load();
  });
  $('srcClear').addEventListener('click', () => {
    $('srcQ').value = ''; q = ''; page = 1; load();
  });
  $('prevBtn').addEventListener('click', () => { if (page > 1) { page--; load(); } });
  $('nextBtn').addEventListener('click', () => { if (page < pages) { page++; load(); } });

  load();
})();
