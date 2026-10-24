/* Textwerkstatt: Frontend ohne Build-Schritt. Kein innerHTML, kein eval, keine externen Requests. */
'use strict';
(() => {
  // ---------------------------------------------------------------- Helfer
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /** DOM-Baukasten: Strings werden immer als Text eingefügt (kein HTML-Parsing). */
  function h(tag, props, ...kids) {
    const n = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v == null || v === false) continue;
        if (k === 'class') n.className = v;
        else if (k === 'text') n.textContent = v;
        else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
        else n.setAttribute(k, v === true ? '' : String(v));
      }
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid);
    return n;
  }

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const nf = new Intl.NumberFormat('de-DE', { maximumFractionDigits: 2 });
  const cpLen = (s) => { let n = 0; for (const _ of s) n++; return n; };
  const countWords = (s) => { const t = s.trim(); return t ? t.split(/\s+/).length : 0; };

  // Unsichtbare / Steuerzeichen, die als Chip sichtbar gemacht werden.
  const INVIS_RE = /[\p{Cf}\p{Cc}\p{Zl}\p{Zp}\p{Zs}\p{Co}͏ᅟᅠ᠋-᠏⠀ㅤ︀-︎ﾠ\u{E0100}-\u{E01EF}]/u;
  function isInvisible(ch) {
    if (ch === ' ' || ch === '\n' || ch === '\t' || ch === '\r' || ch === '️') return false;
    return INVIS_RE.test(ch);
  }
  const cpLabel = (ch) => 'U+' + ch.codePointAt(0).toString(16).toUpperCase().padStart(4, '0');
  const CP_NAMES = {
    0x00A0: 'Geschütztes Leerzeichen', 0x00AD: 'Weiches Trennzeichen', 0x034F: 'Combining Grapheme Joiner',
    0x061C: 'Arabic Letter Mark', 0x180E: 'Mongolian Vowel Separator',
    0x200B: 'Zero Width Space', 0x200C: 'Zero Width Non-Joiner', 0x200D: 'Zero Width Joiner',
    0x200E: 'Left-to-Right Mark', 0x200F: 'Right-to-Left Mark', 0x2028: 'Line Separator', 0x2029: 'Paragraph Separator',
    0x202A: 'LRE', 0x202B: 'RLE', 0x202C: 'PDF', 0x202D: 'LRO', 0x202E: 'RLO', 0x202F: 'Schmales geschütztes Leerzeichen',
    0x2060: 'Word Joiner', 0x2061: 'Function Application', 0x2062: 'Invisible Times', 0x2063: 'Invisible Separator', 0x2064: 'Invisible Plus',
    0x2066: 'LRI', 0x2067: 'RLI', 0x2068: 'FSI', 0x2069: 'PDI', 0x3164: 'Hangul Filler', 0xFEFF: 'BOM / Zero Width No-Break Space',
    0xFFA0: 'Halfwidth Hangul Filler',
  };
  function cpName(ch) {
    const c = ch.codePointAt(0);
    if (CP_NAMES[c]) return CP_NAMES[c];
    if (c >= 0xE0000 && c <= 0xE007F) return 'Tag-Zeichen';
    if (c >= 0xFE00 && c <= 0xFE0F || c >= 0xE0100 && c <= 0xE01EF) return 'Variation Selector';
    if (c >= 0x2000 && c <= 0x200A) return 'Leerzeichen (Sonderbreite)';
    if (c < 0x20 || (c >= 0x7F && c < 0xA0)) return 'Steuerzeichen';
    return 'Unsichtbares Zeichen';
  }
  function chip(ch) {
    return h('span', { class: 'chip', title: `${cpLabel(ch)} ${cpName(ch)}`, 'data-cp': cpLabel(ch), text: `⟦${cpLabel(ch)}⟧` });
  }
  /** Text anhängen; unsichtbare Zeichen werden als Chips dargestellt. */
  function appendText(parent, str) {
    let buf = '';
    for (const ch of str) {
      if (isInvisible(ch)) {
        if (buf) { parent.append(buf); buf = ''; }
        parent.append(chip(ch));
      } else buf += ch;
    }
    if (buf) parent.append(buf);
  }
  const countInvisible = (s) => { let n = 0; for (const ch of s) if (isInvisible(ch)) n++; return n; };

  // ---------------------------------------------------------------- Zustand
  const S = {
    tab: 'scan',
    busy: false,
    abort: null,
    timer: null,
    token: '',
    scan: null,       // { text, data }
    res: null,        // { kind, source, segs, reverted:Set, view, data }
    undoText: null,
    cursor: {},       // codepoint -> zuletzt angesprungener Treffer
    hitEls: new Map(),
    aiEls: new Map(),
    toastTimer: null,
  };

  const input = $('#input');
  const mirror = $('#mirror');
  const results = $('#results');
  const TOKEN_KEY = 'textwerkstatt.token';
  const REMEMBER_KEY = 'textwerkstatt.remember';

  // ---------------------------------------------------------------- Meldungen
  function announce(msg) {
    const sr = $('#sr');
    sr.textContent = '';
    setTimeout(() => { sr.textContent = msg; }, 30);
  }
  function toast(msg, undoFn) {
    const t = $('#toast');
    t.replaceChildren(h('span', { text: msg }));
    if (undoFn) {
      t.append(h('button', { type: 'button', class: 'linkish', text: 'Rückgängig', onclick: () => { undoFn(); t.hidden = true; } }));
    }
    t.hidden = false;
    clearTimeout(S.toastTimer);
    S.toastTimer = setTimeout(() => { t.hidden = true; }, undoFn ? 9000 : 3500);
  }
  function hideAlert() { const a = $('#alert'); a.hidden = true; a.replaceChildren(); }
  function showError(err) {
    const a = $('#alert');
    const parts = [h('strong', { text: err.status === 401 ? 'Nicht autorisiert' : 'Fehler' }), h('span', { text: err.message || 'Unbekannter Fehler.' })];
    if (err.code) parts.push(h('div', {}, 'Fehlercode: ', h('span', { class: 'code', text: err.code })));
    if (err.status === 401) parts.push(h('div', { text: 'Bitte gültigen API-Token im Kopfbereich eintragen.' }));
    a.replaceChildren(...parts);
    a.hidden = false;
    announce('Fehler: ' + (err.message || ''));
  }

  // ---------------------------------------------------------------- API
  class ApiError extends Error {
    constructor(message, code, status) { super(message); this.code = code; this.status = status; }
  }
  async function api(path, { method = 'GET', body, signal } = {}) {
    const headers = { Accept: 'application/json' };
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    if (S.token) headers.Authorization = 'Bearer ' + S.token;
    let resp;
    try {
      resp = await fetch(path, { method, headers, signal, cache: 'no-store', body: body !== undefined ? JSON.stringify(body) : undefined });
    } catch (e) {
      if (e.name === 'AbortError') throw e;
      throw new ApiError('Server nicht erreichbar (' + (e.message || 'Netzwerkfehler') + ').', 'network_error', 0);
    }
    let data = null;
    try { data = await resp.json(); } catch (_) { /* kein JSON */ }
    if (resp.status === 401) throw new ApiError((data && data.error) || 'Token fehlt oder ist ungültig.', (data && data.error_code) || 'unauthorized', 401);
    if (!resp.ok || !data || data.ok === false) {
      throw new ApiError((data && data.error) || `HTTP ${resp.status}`, (data && data.error_code) || null, resp.status);
    }
    return data;
  }

  // ---------------------------------------------------------------- Verbindung / Profile / Modelle
  function setConn(state, text) {
    $('#connDot').dataset.state = state;
    $('#connText').textContent = text;
  }
  function focusToken() { const t = $('#token'); t.focus(); t.select(); }

  async function loadProfiles() {
    const sel = $('#profile');
    setConn('idle', 'Verbindung: prüfe …');
    try {
      const data = await api('/api/llm/profiles');
      const profiles = data.profiles || [];
      sel.replaceChildren(...profiles.map((p) => h('option', { value: p.id, text: `${p.label || p.id}${p.model ? ' (' + p.model + ')' : ''}` })));
      if (!profiles.length) sel.append(h('option', { value: '', text: 'Keine Profile' }));
      const def = data.default && profiles.some((p) => p.id === data.default) ? data.default : (profiles[0] && profiles[0].id);
      if (def) sel.value = def;
      sel.disabled = !profiles.length;
      const cur = profiles.find((p) => p.id === sel.value);
      setConn('ok', cur ? `Verbunden: ${cur.backend || 'LLM'} @ ${cur.base_url_host || 'lokal'}` : 'Verbunden');
      await loadModels();
    } catch (e) {
      sel.disabled = true;
      $('#model').disabled = true;
      if (e.status === 401) {
        setConn('warn', 'Token erforderlich');
        showError(e);
        focusToken();
      } else {
        setConn('err', 'Nicht erreichbar');
        showError(e);
      }
    }
  }
  async function loadModels() {
    const msel = $('#model');
    const profile = $('#profile').value;
    msel.disabled = true;
    try {
      const data = await api('/api/llm/models?profile=' + encodeURIComponent(profile));
      const models = data.models || [];
      const defLabel = data.default_model ? `Standard (${data.default_model})` : 'Standard';
      msel.replaceChildren(h('option', { value: '', text: defLabel }), ...models.map((m) => h('option', { value: m, text: m })));
      msel.value = '';
      msel.disabled = false;
    } catch (e) {
      msel.replaceChildren(h('option', { value: '', text: 'Standard' }));
      msel.disabled = false;
      if (e.status === 401) { setConn('warn', 'Token erforderlich'); showError(e); focusToken(); }
      else setConn('warn', 'Modelle nicht ladbar: ' + e.message);
    }
  }
  function llmParams() {
    const p = {};
    const profile = $('#profile').value;
    const model = $('#model').value;
    if (profile) p.profile = profile;
    if (model) p.model = model;
    return p;
  }

  // ---------------------------------------------------------------- Eingabe, Zähler, Ansicht
  let countRaf = 0;
  function updateCounts() {
    cancelAnimationFrame(countRaf);
    countRaf = requestAnimationFrame(() => {
      const t = input.value;
      $('#counts').textContent = `${nf.format(cpLen(t))} Zeichen · ${nf.format(countWords(t))} Wörter`;
      const inv = countInvisible(t);
      const hint = $('#invisHint');
      hint.hidden = inv === 0;
      hint.textContent = `${nf.format(inv)} unsichtbare${inv === 1 ? 's Zeichen' : ' Zeichen'} – anzeigen`;
      checkStale();
    });
  }
  function checkStale() {
    let stale = false;
    if (S.scan && input.value !== S.scan.text) stale = true;
    if (S.res && input.value !== S.res.source) stale = true;
    const prev = !$('#stale').hidden;
    $('#stale').hidden = !stale;
    if (stale !== prev && !mirror.hidden) refreshMirror();
  }
  function currentView() { return mirror.hidden ? 'edit' : 'mirror'; }
  function showView(view) {
    const m = view === 'mirror';
    if (m) refreshMirror();
    mirror.hidden = !m;
    input.hidden = m;
    $('#viewEdit').setAttribute('aria-pressed', String(!m));
    $('#viewMirror').setAttribute('aria-pressed', String(m));
    if (!m) $('#mirrorInfo').textContent = '';
  }
  function setInput(text, msg) {
    const prev = input.value;
    input.value = text;
    updateCounts();
    if (currentView() === 'mirror') refreshMirror();
    if (msg) toast(msg, () => { input.value = prev; updateCounts(); if (currentView() === 'mirror') refreshMirror(); });
  }

  // Spiegelansicht: Codepoint-Offsets (Python) -> Array.from(text)
  function refreshMirror() {
    const scan = S.scan && S.scan.text === input.value ? S.scan.data : null;
    buildMirror(mirror, input.value, scan);
    if (!scan) $('#mirrorInfo').textContent = '';
  }
  function buildMirror(root, text, scan) {
    root.replaceChildren();
    S.hitEls = new Map();
    S.aiEls = new Map();
    const cps = Array.from(text);
    const hitAt = new Map();
    const aiId = new Int32Array(cps.length);
    const spans = (scan && scan.ai_content && scan.ai_content.spans) || [];
    if (scan) {
      for (const hit of (scan.layer_a && scan.layer_a.hits) || []) {
        for (const o of hit.offsets || []) if (Number.isInteger(o) && o >= 0 && o < cps.length) hitAt.set(o, hit);
      }
      spans.forEach((s, k) => {
        const a = Math.max(0, Number(s.start) | 0);
        const b = Math.min(cps.length, Number(s.end) | 0);
        for (let i = a; i < b; i++) aiId[i] = k + 1;
      });
    }
    let container = root, curAi = 0, buf = '';
    const flush = () => { if (buf) { container.append(buf); buf = ''; } };
    for (let i = 0; i < cps.length; i++) {
      const ch = cps[i];
      if (aiId[i] !== curAi) {
        flush();
        container = root;
        curAi = aiId[i];
        if (curAi) {
          const sp = spans[curAi - 1];
          const w = h('span', { class: 'ai', 'data-span': curAi, title: 'KI-Verdacht' + (sp && sp.reason ? ': ' + sp.reason : '') });
          root.append(w);
          container = w;
          if (!S.aiEls.has(curAi)) S.aiEls.set(curAi, w);
        }
      }
      const hit = hitAt.get(i);
      const inv = isInvisible(ch);
      if (hit) {
        flush();
        const n = inv ? chip(ch) : h('span', { text: ch });
        n.classList.add('hit');
        n.dataset.hit = hit.codepoint;
        n.title = `${hit.codepoint} ${hit.label || ''}`.trim();
        container.append(n);
        if (!S.hitEls.has(hit.codepoint)) S.hitEls.set(hit.codepoint, []);
        S.hitEls.get(hit.codepoint).push(n);
      } else if (inv) {
        flush();
        container.append(chip(ch));
      } else buf += ch;
    }
    flush();
    if (!cps.length) root.append(h('span', { class: 'muted', text: 'Kein Text.' }));
  }
  function jumpToHit(cp) {
    showView('mirror');
    const els = S.hitEls.get(cp);
    if (!els || !els.length) { $('#mirrorInfo').textContent = `${cp}: keine Treffer in der Ansicht`; return; }
    const i = ((S.cursor[cp] ?? -1) + 1) % els.length;
    S.cursor[cp] = i;
    $$('.hit-active, .hit-current', mirror).forEach((n) => n.classList.remove('hit-active', 'hit-current'));
    els.forEach((e) => e.classList.add('hit-active'));
    els[i].classList.add('hit-current');
    els[i].scrollIntoView({ block: 'center', inline: 'nearest', behavior: reduceMotion ? 'auto' : 'smooth' });
    const msg = `${cp}: Treffer ${i + 1} von ${els.length}`;
    $('#mirrorInfo').textContent = msg;
    announce(msg);
  }
  function jumpToAi(id) {
    showView('mirror');
    const el = S.aiEls.get(id);
    if (!el) return;
    $$('.ai-current', mirror).forEach((n) => n.classList.remove('ai-current'));
    el.classList.add('ai-current');
    el.scrollIntoView({ block: 'center', behavior: reduceMotion ? 'auto' : 'smooth' });
  }

  // ---------------------------------------------------------------- Tabs
  function selectTab(name, focus) {
    S.tab = name;
    for (const t of $$('[role="tab"]')) {
      const on = t.id === 'tab-' + name;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      $('#' + t.getAttribute('aria-controls')).hidden = !on;
      if (on && focus) t.focus();
    }
  }
  function initTabs() {
    const tabs = $$('[role="tab"]');
    tabs.forEach((t, i) => {
      t.addEventListener('click', () => selectTab(t.id.slice(4)));
      t.addEventListener('keydown', (e) => {
        let j = null;
        if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
        else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === 'Home') j = 0;
        else if (e.key === 'End') j = tabs.length - 1;
        if (j !== null) { e.preventDefault(); selectTab(tabs[j].id.slice(4), true); }
      });
    });
  }

  // ---------------------------------------------------------------- Ausführen mit Ladezustand
  function setBusy(on, label) {
    S.busy = on;
    $$('[data-run]').forEach((b) => { b.disabled = on; });
    $('#cancel').disabled = !on;
    $('#runbar').classList.toggle('busy', on);
    results.setAttribute('aria-busy', String(on));
    clearInterval(S.timer);
    const txt = $('#runText');
    if (on) {
      const t0 = Date.now();
      const tick = () => {
        const s = Math.floor((Date.now() - t0) / 1000);
        txt.textContent = `${label} läuft … ${s} s` + (s >= 10 ? ' (LLM-Aufrufe können mehrere Minuten dauern)' : '');
      };
      tick();
      S.timer = setInterval(tick, 500);
      announce(label + ' gestartet.');
    }
  }
  async function run(label, fn) {
    if (S.busy) return;
    hideAlert();
    const ac = new AbortController();
    S.abort = ac;
    setBusy(true, label);
    $('#runbar').scrollIntoView({ block: 'nearest' });
    let outcome = 'Fertig.';
    try {
      await fn(ac.signal);
      announce(label + ' abgeschlossen.');
    } catch (e) {
      if (e.name === 'AbortError') outcome = 'Abgebrochen.';
      else {
        outcome = 'Fehlgeschlagen.';
        showError(e);
        if (e.status === 401) { setConn('warn', 'Token erforderlich'); focusToken(); }
      }
    } finally {
      S.abort = null;
      setBusy(false);
      $('#runText').textContent = outcome;
      // einspaltig: Ergebnis nach oben holen, sonst bleibt es unterhalb der Optionen unsichtbar
      if (outcome === 'Fertig.' && window.matchMedia('(max-width: 61.99rem)').matches) {
        $('#h-out').scrollIntoView({ block: 'start', behavior: reduceMotion ? 'auto' : 'smooth' });
      }
    }
  }

  function requireText() {
    if (!input.value.trim()) {
      toast('Bitte zuerst einen Text eingeben.');
      showView('edit');
      input.focus();
      return false;
    }
    return true;
  }

  // ---------------------------------------------------------------- Scan
  function runScan() {
    if (!requireText()) return;
    const text = input.value;
    const body = { text, ai_detect: $('#aiDetect').value, ...llmParams() };
    run('Scan', async (signal) => {
      const data = await api('/api/text/scan', { method: 'POST', body, signal });
      S.res = null;
      S.scan = { text, data };
      S.cursor = {};
      renderScan(data);
      $('#stale').hidden = true;
      showView('mirror');
      buildMirror(mirror, text, data);
    });
  }

  const VERDICT = {
    likely_ai: ['Wahrscheinlich KI-generiert', 'bad'],
    mixed: ['Gemischt / unklar', 'warn'],
    likely_human: ['Wahrscheinlich menschlich', 'ok'],
    unknown: ['Unbekannt', 'muted'],
  };
  function fmtConf(v) {
    if (typeof v === 'number') return v <= 1 ? `${Math.round(v * 100)} %` : nf.format(v);
    return v == null ? '–' : String(v);
  }
  function fmtScore(v) { return typeof v === 'number' ? nf.format(Math.round(v * 100) / 100) : '–'; }
  function meter(v, small, label) {
    const pct = typeof v === 'number' ? Math.max(0, Math.min(1, v)) * 100 : 0;
    return h('span', {
      class: 'meter' + (small ? ' small' : ''), role: 'meter', 'aria-label': label,
      'aria-valuemin': 0, 'aria-valuemax': 1, 'aria-valuenow': typeof v === 'number' ? v : null,
      'aria-valuetext': typeof v === 'number' ? fmtScore(v) : 'kein Wert',
    }, h('i', { style: `width:${pct.toFixed(1)}%` }));
  }
  function kvText(obj) {
    if (obj == null) return '';
    if (typeof obj !== 'object') return String(obj);
    return Object.entries(obj).slice(0, 8).map(([k, v]) => `${k}: ${typeof v === 'object' && v !== null ? JSON.stringify(v).slice(0, 80) : v}`).join(' · ');
  }

  function renderScan(d) {
    const la = d.layer_a || {}, lb = d.layer_b || {}, ai = d.ai_content || {};
    const hits = la.hits || [];
    const out = [];

    out.push(h('div', { class: 'kvline' }, 'Länge: ', h('b', { text: nf.format(d.length ?? cpLen(S.scan.text)) + ' Zeichen' }), ' · Layer A: ', h('b', { text: `${nf.format(la.suspicious_total ?? 0)} verdächtige Zeichen` }), ' · Treffer-Arten: ', h('b', { text: String(hits.length) })));

    // Legende
    out.splice(1, 0, h('div', { class: 'legend' },
      h('span', {}, h('i', { class: 'sw hit' }), 'Layer-A-Treffer'),
      h('span', {}, h('i', { class: 'sw ai' }), 'KI-Verdacht'),
      h('span', {}, h('span', { class: 'chip', text: '⟦U+200B⟧' }), 'unsichtbares Zeichen')));

    // Layer A
    const a = h('section', { class: 'card', 'aria-labelledby': 'h-la' }, h('h3', { id: 'h-la' }, 'Layer A: unsichtbare und verdächtige Zeichen'));
    if (hits.length) {
      const tbody = h('tbody');
      for (const hit of hits) {
        const cp = hit.codepoint;
        const tr = h('tr', { class: 'row', onclick: () => jumpToHit(cp) },
          h('td', {}, h('button', { type: 'button', class: 'linkish', 'aria-label': `${cp} im Text anzeigen`, text: cp, onclick: (e) => { e.stopPropagation(); jumpToHit(cp); } })),
          h('td', { text: hit.label || '' }),
          h('td', {}, h('code', { text: hit.kind || '' })),
          h('td', { text: fmtConf(hit.confidence) }),
          h('td', { class: 'num', text: nf.format(hit.count ?? (hit.offsets || []).length) }));
        tbody.append(tr);
      }
      a.append(h('div', { class: 'tablewrap' }, h('table', { class: 'tbl' },
        h('caption', { class: 'sr-only', text: 'Layer-A-Treffer; Zeile anklicken, um im Text zu springen' }),
        h('thead', {}, h('tr', {}, ...['Codepoint', 'Bezeichnung', 'Art', 'Konfidenz'].map((t) => h('th', { scope: 'col', text: t })), h('th', { scope: 'col', class: 'num', text: 'Anzahl' }))),
        tbody)));
      a.append(h('p', { class: 'hint', text: 'Zeile anklicken: springt zum (nächsten) Treffer in der markierten Eingabe.' }));
    } else {
      a.append(h('p', { class: 'muted', text: 'Keine verdächtigen Zeichen gefunden.' }));
    }
    if ((la.notes || []).length) a.append(h('ul', { class: 'notes' }, ...la.notes.map((n) => h('li', { text: n }))));
    out.push(a);

    // Layer B
    const b = h('section', { class: 'card', 'aria-labelledby': 'h-lb' }, h('h3', { id: 'h-lb' }, 'Layer B: statistische Detektoren'));
    const dets = lb.detectors || [];
    if (dets.length) {
      b.append(h('ul', { class: 'list' }, ...dets.map((dt) => h('li', {},
        h('span', { class: 'name', text: dt.name }),
        h('span', { class: 'badge ' + (dt.available ? 'ok' : 'muted'), text: dt.available ? 'verfügbar' : 'nicht verfügbar' }),
        dt.detail ? h('span', { class: 'kv', text: kvText(dt.detail) }) : null))));
    } else b.append(h('p', { class: 'muted', text: 'Keine Detektoren gemeldet.' }));
    if (lb.stylometry) {
      const st = lb.stylometry;
      b.append(h('div', { class: 'kvline' }, 'Stilometrie: Score ', h('b', { text: fmtScore(st.score) }), ' · Konfidenz ', h('b', { text: String(st.confidence_level ?? '–') }), ' · Dichte ', h('b', { text: String(st.density_tier ?? '–') })));
    }
    out.push(b);

    // KI-Inhalt
    const [vText, vCls] = VERDICT[ai.verdict] || VERDICT.unknown;
    const c = h('section', { class: 'card', 'aria-labelledby': 'h-ai' }, h('h3', { id: 'h-ai' }, 'KI-generierter Inhalt'));
    c.append(h('div', { class: 'verdict' },
      h('span', { class: 'badge ' + vCls, text: vText }),
      meter(ai.score, false, 'KI-Score'),
      h('span', { class: 'kv' }, 'Score ', h('b', { text: fmtScore(ai.score) }), ' (0 = menschlich, 1 = KI)')));
    for (const m of ai.methods || []) {
      c.append(h('div', { class: 'method' },
        h('span', { class: 'name', text: m.name }),
        h('span', { class: 'm-score' }, meter(m.score, true, `Score ${m.name}`), h('span', { text: fmtScore(m.score) })),
        m.detail ? h('span', { class: 'm-detail', text: m.detail }) : null,
        m.error ? h('span', { class: 'm-err', text: 'Fehler: ' + m.error }) : null));
    }
    const spans = ai.spans || [];
    if (spans.length) {
      c.append(h('p', { class: 'hint', text: `${spans.length} auffällige Abschnitte (in der markierten Eingabe violett unterlegt):` }));
      c.append(h('div', { class: 'spanlist' }, ...spans.slice(0, 40).map((s, k) => h('button', {
        type: 'button', class: 'btn sm', title: s.reason || '', onclick: () => jumpToAi(k + 1),
        text: `${k + 1}: Zeichen ${s.start}–${s.end}${s.reason ? ' · ' + String(s.reason).slice(0, 28) : ''}`,
      }))));
    }
    c.append(h('p', { class: 'disclaimer', text: ai.disclaimer || 'KI-Erkennung ist unzuverlässig und liefert nur Indizien, keine Beweise.' }));
    out.push(c);

    results.replaceChildren(...out);
  }

  // ---------------------------------------------------------------- Diff-Ergebnisse
  const OP_LABEL = { delete: 'Gelöscht', insert: 'Eingefügt', replace: 'Ersetzt' };

  function finalText(res) {
    return res.segs.map((s, i) => (s.op === 'equal' || res.reverted.has(i) ? s.a : s.b)).join('');
  }
  function computeStats(res) {
    let changes = 0, active = 0, words = 0, rem = 0, add = 0;
    res.segs.forEach((s, i) => {
      if (s.op === 'equal') return;
      changes++;
      if (res.reverted.has(i)) return;
      active++;
      words += Math.max(countWords(s.a), countWords(s.b));
      rem += cpLen(s.a);
      add += cpLen(s.b);
    });
    const src = Math.max(1, countWords(res.source));
    return { changes, active, pct: Math.min(100, Math.round((words / src) * 100)), rem, add };
  }

  function fillChange(node, seg, reverted, idx) {
    node.replaceChildren();
    node.className = `chg ${seg.op}${reverted ? ' reverted' : ''}`;
    node.setAttribute('aria-pressed', String(!reverted));
    node.setAttribute('title', reverted ? 'Zurückgesetzt – klicken, um die Änderung wieder anzuwenden' : 'Übernommen – klicken, um die Änderung zurückzusetzen');
    node.append(h('span', { class: 'sr-only', text: `Änderung ${idx + 1}, ${OP_LABEL[seg.op] || seg.op}, ${reverted ? 'zurückgesetzt' : 'übernommen'}: ` }));
    const part = (cls, text) => { const s = h('span', { class: cls }); appendText(s, text); return s; };
    if (!reverted) {
      if (seg.a) node.append(part('del', seg.a));
      if (seg.a && seg.b) node.append(h('span', { class: 'arrow', 'aria-hidden': 'true', text: '→' }));
      if (seg.b) node.append(part('ins', seg.b));
    } else {
      if (seg.a) node.append(part('keep', seg.a));
      if (seg.b) node.append(part('ghost', seg.b));
    }
  }

  /** Baut einen Diff-Container. interactive=false: nur Anzeige. */
  function buildDiff(segs, { interactive, reverted }) {
    const box = h('div', { class: 'diff' + (interactive ? '' : ' ro'), tabindex: '0', role: 'group', 'aria-label': 'Diff: Änderungen am Text' });
    const nodes = new Map();
    segs.forEach((s, i) => {
      if (s.op === 'equal') { appendText(box, s.a); return; }
      const n = h('span', interactive ? { role: 'button', tabindex: '0', 'data-idx': i } : { 'data-idx': i });
      fillChange(n, s, reverted ? reverted.has(i) : false, i);
      box.append(n);
      nodes.set(i, n);
    });
    if (!segs.length) box.append(h('span', { class: 'muted', text: 'Keine Änderungen.' }));
    return { box, nodes };
  }

  let ui = null; // Referenzen des aktuellen Diff-Ergebnisses

  function renderEdit(kind, data, segs) {
    S.scan = null;
    S.res = { kind, source: data.original, segs: segs.map((s) => ({ op: s.op, a: s.a ?? '', b: s.b ?? '' })), reverted: new Set(), view: 'total', data };
    S.cursor = {};
    buildEditUI();
    $('#stale').hidden = input.value === S.res.source;
    if (!mirror.hidden) refreshMirror();
  }

  function buildEditUI() {
    const res = S.res, d = res.data;
    const out = [];
    const statsEl = h('dl', { class: 'stats' });
    const info = h('div');
    const card = h('section', { class: 'card', 'aria-labelledby': 'h-res' },
      h('h3', { id: 'h-res' }, res.kind === 'clean' ? 'Bereinigung' : 'Paraphrase', h('span', { class: 'sub', id: 'changeSub' })),
      statsEl);

    if (res.kind === 'paraphrase' && d.noop) {
      card.prepend(h('div', { class: 'note' }, 'Das Modell hat den Text unverändert zurückgegeben. Versuche eine höhere Intensität, einen anderen Modus oder ein anderes Modell.'));
    }

    const tools = h('div', { class: 'toolbar' });
    if (res.kind === 'clean') {
      const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Diff-Ansicht' },
        h('button', { type: 'button', id: 'vTotal', 'aria-pressed': 'true', text: 'Gesamt', onclick: () => setView('total') }),
        h('button', { type: 'button', id: 'vSplit', 'aria-pressed': 'false', text: 'Layer A / Layer B getrennt', onclick: () => setView('split') }));
      tools.append(seg);
    }
    tools.append(h('span', { class: 'grow' }));
    tools.append(h('button', { type: 'button', class: 'btn sm', id: 'accAll', text: 'Alle übernehmen', onclick: () => setAll(false) }));
    tools.append(h('button', { type: 'button', class: 'btn sm', id: 'revAll', text: 'Alle zurücksetzen', onclick: () => setAll(true) }));
    card.append(tools);
    card.append(h('div', { class: 'legend' },
      h('span', {}, h('i', { class: 'sw del' }), 'entfernt'),
      h('span', {}, h('i', { class: 'sw ins' }), 'eingefügt'),
      h('span', {}, h('span', { class: 'chip', text: '⟦U+200B⟧' }), 'unsichtbares Zeichen'),
      h('span', { class: 'muted', text: 'Änderung anklicken oder per Enter/Leertaste zurücksetzen bzw. erneut anwenden.' })));

    const diffHost = h('div', { id: 'diffHost' });
    card.append(diffHost);
    card.append(info);
    out.push(card);

    // Endtext
    const fin = h('div', { class: 'final', id: 'finalText', tabindex: '0', role: 'region', 'aria-label': 'Endtext' });
    const finCard = h('section', { class: 'card', 'aria-labelledby': 'h-final' },
      h('h3', { id: 'h-final' }, 'Endtext', h('span', { class: 'sub', id: 'finalMeta' })),
      h('div', { class: 'toolbar' },
        h('button', { type: 'button', class: 'btn primary sm', id: 'copyBtn', text: 'Kopieren', onclick: copyFinal }),
        h('button', { type: 'button', class: 'btn sm', id: 'adoptBtn', text: 'Als Eingabe übernehmen', onclick: adoptFinal })),
      fin);
    out.push(finCard);

    // Zusatzinfos
    info.append(...buildInfo(res));

    results.replaceChildren(...out);
    ui = { statsEl, diffHost, fin, nodes: new Map(), diffBox: null };
    renderDiffView();
    updateSummary();
  }

  function buildInfo(res) {
    const d = res.data;
    const rows = [];
    const lab = (obj) => Object.entries(obj || {}).map(([k, v]) => `${k} ×${v}`).join(', ');
    if (res.kind === 'clean') {
      const la = d.layer_a || {};
      rows.push(h('div', { class: 'kvline' }, 'Layer A: ', h('b', { text: `${nf.format(la.removed_count ?? 0)} entfernt` }), ' · ', h('b', { text: `${nf.format(la.replaced_count ?? 0)} ersetzt` }),
        Object.keys(la.removed || {}).length ? h('span', { text: ' · entfernt: ' + lab(la.removed) }) : null,
        Object.keys(la.replaced || {}).length ? h('span', { text: ' · ersetzt: ' + lab(la.replaced) }) : null));
      const lb = d.layer_b || {};
      if (lb.applied) {
        rows.push(h('div', { class: 'kvline' }, 'Layer B: ', h('b', { text: 'angewendet' }),
          lb.strategy ? h('span', { class: 'tag', text: lb.strategy }) : null,
          lb.model ? h('span', {}, 'Modell ', h('b', { text: lb.model })) : null,
          lb.profile ? h('span', {}, 'Profil ', h('b', { text: lb.profile })) : null,
          Array.isArray(lb.steps) ? h('span', { text: `${lb.steps.length} Schritt${lb.steps.length === 1 ? '' : 'e'}` }) : null));
      } else {
        rows.push(h('div', { class: 'kvline' }, 'Layer B: ', h('b', { text: 'nicht angewendet' }), lb.reason ? h('span', { text: '(' + lb.reason + ')' }) : null));
      }
    } else {
      const la = d.layer_a || {};
      rows.push(h('div', { class: 'kvline' }, 'Modus ', h('b', { text: String(d.mode ?? '–') }), ' · Intensität ', h('b', { text: `${Math.round((d.intensity ?? 0) * 100)} %` }),
        d.model ? h('span', {}, ' · Modell ', h('b', { text: d.model })) : null,
        d.profile ? h('span', {}, ' · Profil ', h('b', { text: d.profile })) : null,
        d.chunks != null ? h('span', { text: ` · ${d.chunks} Abschnitt${d.chunks === 1 ? '' : 'e'}` }) : null));
      if (d.layer_a) rows.push(h('div', { class: 'kvline' }, 'Layer A danach: ', h('b', { text: `${nf.format(la.removed_count ?? 0)} entfernt` }), ' · ', h('b', { text: `${nf.format(la.replaced_count ?? 0)} ersetzt` })));
    }
    return rows;
  }

  function renderDiffView() {
    const res = S.res;
    const host = ui.diffHost;
    if (res.kind === 'clean') {
      $('#vTotal').setAttribute('aria-pressed', String(res.view === 'total'));
      $('#vSplit').setAttribute('aria-pressed', String(res.view === 'split'));
    }
    $('#accAll').disabled = $('#revAll').disabled = res.view === 'split';
    if (res.view === 'total') {
      const { box, nodes } = buildDiff(res.segs, { interactive: true, reverted: res.reverted });
      ui.nodes = nodes;
      ui.diffBox = box;
      box.addEventListener('click', onDiffClick);
      box.addEventListener('keydown', onDiffKey);
      host.replaceChildren(box);
    } else {
      const d = res.data;
      const norm = (arr) => (arr || []).map((s) => ({ op: s.op, a: s.a ?? '', b: s.b ?? '' }));
      const A = buildDiff(norm(d.diff_a), { interactive: false });
      const B = buildDiff(norm(d.diff_b), { interactive: false });
      ui.nodes = new Map();
      host.replaceChildren(
        h('p', { class: 'diff-title', text: 'Layer A: Original → nach Layer A' }), A.box,
        h('p', { class: 'diff-title', text: 'Layer B: nach Layer A → Endtext' }), B.box,
        h('p', { class: 'hint', text: 'Die getrennte Ansicht ist nur zur Ansicht. Einzelne Änderungen lassen sich in der Gesamtansicht zurücksetzen.' }));
    }
  }
  function setView(v) {
    if (!S.res) return;
    S.res = { ...S.res, view: v };
    renderDiffView();
  }
  function onDiffClick(e) {
    const n = e.target.closest('[data-idx]');
    if (n && ui.diffBox.contains(n)) toggleChange(Number(n.dataset.idx));
  }
  function onDiffKey(e) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const n = e.target.closest('[data-idx]');
    if (n && n === e.target) { e.preventDefault(); toggleChange(Number(n.dataset.idx)); }
  }
  function toggleChange(idx) {
    const next = new Set(S.res.reverted);
    if (next.has(idx)) next.delete(idx); else next.add(idx);
    S.res = { ...S.res, reverted: next };
    const node = ui.nodes.get(idx);
    if (node) fillChange(node, S.res.segs[idx], next.has(idx), idx);
    updateSummary();
    announce(next.has(idx) ? 'Änderung zurückgesetzt.' : 'Änderung übernommen.');
  }
  function setAll(revert) {
    const next = new Set();
    if (revert) S.res.segs.forEach((s, i) => { if (s.op !== 'equal') next.add(i); });
    S.res = { ...S.res, reverted: next };
    ui.nodes.forEach((node, idx) => fillChange(node, S.res.segs[idx], next.has(idx), idx));
    updateSummary();
  }
  function updateSummary() {
    const res = S.res;
    const st = computeStats(res);
    const stat = (label, value) => h('div', { class: 'stat' }, h('dt', { text: label }), h('dd', { text: value }));
    ui.statsEl.replaceChildren(
      stat('Änderungen aktiv', `${st.active} / ${st.changes}`),
      stat('Wörter geändert', `${st.pct} %`),
      stat('Zeichen entfernt', `−${nf.format(st.rem)}`),
      stat('Zeichen eingefügt', `+${nf.format(st.add)}`));
    const final = finalText(res);
    ui.fin.replaceChildren();
    appendText(ui.fin, final);
    if (!final) ui.fin.append(h('span', { class: 'muted', text: '(leer)' }));
    const inv = countInvisible(final);
    $('#finalMeta').textContent = `${nf.format(cpLen(final))} Zeichen · ${nf.format(countWords(final))} Wörter` + (inv ? ` · ${inv} unsichtbare Zeichen verbleiben` : '');
    $('#copyBtn').disabled = $('#adoptBtn').disabled = false;
  }

  async function copyFinal() {
    const text = finalText(S.res);
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const ta = h('textarea', { 'aria-hidden': 'true', style: 'position:fixed;opacity:0;top:0;left:0' });
      ta.value = text;
      document.body.append(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (__) { ok = false; }
      ta.remove();
      if (!ok) { toast('Kopieren nicht möglich. Bitte Endtext markieren und manuell kopieren.'); return; }
    }
    toast('Endtext kopiert.');
  }
  function adoptFinal() {
    const text = finalText(S.res);
    setInput(text, 'Endtext als Eingabe übernommen.');
    showView('edit');
    checkStale();
  }

  // ---------------------------------------------------------------- Bereinigen / Paraphrasieren
  function runClean() {
    if (!requireText()) return;
    const body = {
      text: input.value,
      ...llmParams(),
      layer_a: {
        nfkc: $('#a-nfkc').checked,
        aggressive_homoglyphs: $('#a-homoglyphs').checked,
        normalize_spaces: $('#a-spaces').checked,
        strip_emoji_glue: $('#a-emoji').checked,
        strip_bidi: $('#a-bidi').checked,
      },
      layer_b: { enabled: $('#b-enable').checked },
    };
    if (body.layer_b.enabled) {
      const strat = $('#b-strategy').value.trim();
      const style = $('#b-style').value.trim();
      if (strat) body.layer_b.strategy = strat;
      if (style) body.layer_b.style = style;
    }
    run(body.layer_b.enabled ? 'Bereinigung (Layer A + B)' : 'Bereinigung', async (signal) => {
      const data = await api('/api/text/clean', { method: 'POST', body, signal });
      renderEdit('clean', data, data.diff || []);
    });
  }
  function runParaphrase() {
    if (!requireText()) return;
    const mode = ($('input[name="mode"]:checked') || {}).value || 'standard';
    const body = {
      text: input.value, mode,
      intensity: Number($('#intensity').value) / 100,
      layer_a_after: $('#layerAAfter').checked,
      ...llmParams(),
    };
    run('Paraphrase', async (signal) => {
      const data = await api('/api/text/paraphrase', { method: 'POST', body, signal });
      renderEdit('paraphrase', data, data.diff || []);
    });
  }
  function startCurrent() {
    if (S.busy) return;
    ({ scan: runScan, clean: runClean, paraphrase: runParaphrase })[S.tab]();
  }

  // ---------------------------------------------------------------- Beispieltext
  const SAMPLE = [
    'Sehr geehrte Damen und Herren,',
    '',
    'wir freuen uns, Ihnen mitteilen zu dürfen, dass das Projekt 😀 erfolgreich abgeschlossen wurde.​ Die Ergebnisse sind ​sehr wichtig für die weitere Planung, und wir⁠ möchten betonen, dass die Zusammenarbeit außerordentlich konstruktiv war.‍',
    '',
    'Bitte beachten Sie: Dieser Text enth­hält ein paar unsichtbare Zeichen, ein kyrillisches „а“ in Wаsser und zwei Tag-Zeichen 󠁁󠁂 am Ende.﻿',
  ].join('\n');

  // ---------------------------------------------------------------- Initialisierung
  function init() {
    // Token
    try {
      if (localStorage.getItem(REMEMBER_KEY) === '1') {
        $('#remember').checked = true;
        S.token = localStorage.getItem(TOKEN_KEY) || '';
        $('#token').value = S.token;
      }
    } catch (_) { /* Speicher gesperrt */ }
    const persistToken = () => {
      try {
        if ($('#remember').checked) { localStorage.setItem(TOKEN_KEY, S.token); localStorage.setItem(REMEMBER_KEY, '1'); }
        else { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(REMEMBER_KEY); }
      } catch (_) { /* ignorieren */ }
    };
    $('#token').addEventListener('input', (e) => { S.token = e.target.value.trim(); persistToken(); });
    $('#token').addEventListener('change', () => loadProfiles());
    $('#remember').addEventListener('change', persistToken);
    $('#connForm').addEventListener('submit', (e) => { e.preventDefault(); loadProfiles(); });
    $('#profile').addEventListener('change', () => { loadModels(); });
    $('#refresh').addEventListener('click', () => { hideAlert(); loadProfiles(); });

    // Eingabe
    input.addEventListener('input', updateCounts);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); startCurrent(); }
    });
    $('#viewEdit').addEventListener('click', () => { showView('edit'); input.focus(); });
    $('#viewMirror').addEventListener('click', () => showView('mirror'));
    $('#invisHint').addEventListener('click', () => showView('mirror'));
    $('#paste').addEventListener('click', async () => {
      try {
        const t = await navigator.clipboard.readText();
        showView('edit');
        input.focus();
        input.setRangeText(t, input.selectionStart, input.selectionEnd, 'end');
        updateCounts();
      } catch (_) {
        showView('edit');
        input.focus();
        toast('Zugriff auf die Zwischenablage nicht erlaubt. Bitte mit Strg/Cmd + V einfügen.');
      }
    });
    $('#clear').addEventListener('click', () => {
      if (!input.value) { input.focus(); return; }
      setInput('', 'Eingabe geleert.');
      showView('edit');
      input.focus();
    });
    $('#sample').addEventListener('click', () => {
      setInput(SAMPLE, input.value ? 'Beispieltext geladen.' : null);
      showView('edit');
    });

    // Tabs und Optionen
    initTabs();
    $$('[data-run]').forEach((b) => b.addEventListener('click', startCurrent));
    $('#cancel').addEventListener('click', () => { if (S.abort) S.abort.abort(); });
    $('#b-enable').addEventListener('change', (e) => {
      $('#b-strategy').disabled = $('#b-style').disabled = !e.target.checked;
    });
    $('#intensity').addEventListener('input', (e) => { $('#intensityOut').textContent = e.target.value + ' %'; });

    updateCounts();
    loadProfiles();
  }
  init();
})();
