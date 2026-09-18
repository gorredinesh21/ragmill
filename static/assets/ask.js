/* RAGMill ask page — the working demo: Search (no LLM) + Ask (SSE pipeline).
   Ported unchanged from the original single-page demo: SSE stage rendering
   (retrieve -> rerank -> answer with [n] citations), red error boxes and a
   hard 90s client timeout. The stream ALWAYS terminates (done | error). */
const $ = id => document.getElementById(id);
const ASK_TIMEOUT_MS = 90000;   // hard client-side deadline for the ask stream
let askSeq = 0;                 // ignore stale streams after a new question

function esc(s){ const d=document.createElement('div'); d.textContent=s??''; return d.innerHTML; }

function showError(title, msg){
  const el = document.createElement('div');
  el.className = 'error-box';
  el.innerHTML = `<b>${esc(title)}</b>${esc(msg)}`;
  $('out').appendChild(el);
}

function setBusy(b){
  $('askBtn').disabled = b; $('searchBtn').disabled = b;
  if(b){ $('meta').textContent = 'streaming…'; }
}

function useChip(btn){ $('q').value = btn.textContent.trim(); doAsk(); }

/* ======================= search (no LLM) ======================= */

async function doSearch(){
  const q = $('q').value.trim(); if(!q) return;
  $('out').innerHTML = ''; $('meta').textContent = 'searching…';
  setBusy(true);
  try{
    const r = await fetch('/api/search', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query:q, k:8})});
    if(!r.ok){
      let msg = 'HTTP ' + r.status;
      try{ const b = await r.json(); if(b.detail) msg = b.detail; }catch(e){}
      throw new Error(msg);
    }
    const b = await r.json();
    $('meta').textContent =
      `hybrid search — ${b.took_ms} ms · dense top: ${b.dense_top3.join(', ')} · sparse top: ${b.sparse_top3.join(', ')}`;
    if(!b.fused.length){
      showError('No results', 'Retrieval returned 0 documents — try different wording.');
      return;
    }
    const box = document.createElement('div');
    box.className = 'stage retrieve';
    box.innerHTML = '<h3>hybrid search — top matches</h3>' + b.fused.map((p,i) =>
      `<div class="paper"><span class="id">${esc(p.arxiv_id)}</span> ·
       <span class="score">rrf ${p.rrf_score}</span><br><b>${i+1}. ${esc(p.title)}</b>
       <div class="detail">${esc((p.categories||[]).join(' · '))}</div></div>`).join('');
    $('out').appendChild(box);
  }catch(e){
    showError('Search failed', String(e.message||e));
  }finally{
    setBusy(false);
  }
}

/* ======================= ask (SSE, always terminates) ======================= */

async function doAsk(){
  const q = $('q').value.trim(); if(!q) return;
  const seq = ++askSeq;
  $('out').innerHTML = '';
  setBusy(true);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ASK_TIMEOUT_MS);
  let terminal = false;   // became true on `done` or `error`

  const fail = (title, msg) => { terminal = true; showError(title, msg); };

  try{
    const r = await fetch('/api/query', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({query:q}), signal: ctrl.signal});
    if(!r.ok || !r.body){
      let msg = 'HTTP ' + r.status;
      try{ const b = await r.json(); if(b.detail) msg = b.detail; }catch(e){}
      throw new Error(msg);
    }
    const reader = r.body.getReader(); const dec = new TextDecoder();
    let buf='';
    while(!terminal){
      const {done, value} = await reader.read(); if(done) break;
      buf += dec.decode(value, {stream:true});
      let idx;
      while((idx = buf.indexOf('\n\n')) >= 0){
        const chunk = buf.slice(0, idx); buf = buf.slice(idx+2);
        let ev='message', data='';
        for(const line of chunk.split('\n')){
          if(line.startsWith('event:')) ev = line.slice(6).trim();
          if(line.startsWith('data:')) data += line.slice(5).trim();
        }
        if(!data) continue;
        try { handle(ev, JSON.parse(data)); } catch(e){}
        if(terminal) break;
      }
    }
    if(!terminal && seq === askSeq){
      fail('Stream ended without an answer',
           'The server closed the stream before answering. Please try again.');
    }
  }catch(e){
    if(seq !== askSeq) return;
    if(e.name === 'AbortError'){
      fail('Timed out after 90 seconds',
           'No answer arrived within 90s — the service may still be waking up. Try again.');
    }else{
      fail('Ask failed', String(e.message||e));
    }
  }finally{
    clearTimeout(timer);
    if(seq === askSeq) setBusy(false);
  }

  /* ---- render one SSE event (ignored if a newer question took over) ---- */
  function handle(ev, d){
    if(seq !== askSeq) return;
    if(ev === 'error'){
      fail('Error — ' + (d.code || 'unknown'), d.message || 'unknown error');
      return;
    }
    if(ev !== 'stage') {
      if(ev === 'done'){
        terminal = true;
        const t = d.timings || {};
        $('meta').textContent =
          `done — retrieve ${t.retrieve_ms ?? '?'} ms · rerank ${t.rerank_ms ?? '?'} ms · answer ${t.answer_ms ?? '?'} ms`;
      }
      return;
    }
    if(d.stage === 'retrieve'){
      const box = document.createElement('div');
      box.className = 'stage retrieve';
      box.innerHTML = `<h3>① retrieved passages — ${d.took_ms} ms</h3>` +
        d.fused_top.map((p,i) =>
          `<div class="paper"><span class="id">${esc(p.arxiv_id)}</span>
           <span class="score">rrf ${p.rrf_score}</span><br>
           <b>${i+1}. ${esc(p.title)}</b>
           <div class="detail">${esc((p.categories||[]).join(' · '))}</div></div>`).join('');
      $('out').appendChild(box);
    }
    if(d.stage === 'rerank'){
      const box = document.createElement('div');
      box.className = 'stage rerank';
      box.innerHTML = `<h3>② reranked — ${esc(d.reranker)} — ${d.took_ms} ms</h3>` +
        d.top.map((p,i) =>
          `<div class="paper"><span class="id">${esc(p.arxiv_id)}</span>
           <span class="score">${p.rerank_score ?? ''}</span><br>
           <b>${i+1}. ${esc(p.title)}</b></div>`).join('');
      $('out').appendChild(box);
    }
    if(d.stage === 'answer'){
      const box = document.createElement('div');
      box.className = 'stage answer';
      const linkedAnswer = esc(d.answer)
        .replace(/\[(\d+)\]/g,
          (m,n) => `<a class="citem" href="#src-${n}">${m}</a>`);
      const sources = (d.citations || []).map(c =>
        `<div class="src" id="src-${c.n}"><span class="n">[${c.n}]</span>
         <a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)}</a>
         <span class="score"> · ${esc(c.arxiv_id)}</span></div>`).join('');
      box.innerHTML = `<h3>③ answer — ${esc(d.llm)} — ${d.took_ms} ms</h3>
        <div class="answer">${linkedAnswer}</div>
        <div class="sources"><h4>Sources — click to verify</h4>${sources}</div>`;
      $('out').appendChild(box);
    }
  }
}

/* Enter key runs Ask */
$('q').addEventListener('keydown', e => { if(e.key === 'Enter') doAsk(); });
