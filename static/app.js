/* Puzzle BI-бот — клиентская часть чата и дашбордов (без сборки, чистый JS). */
(() => {
  'use strict';

  // ------------------------------------------------------------------ утилиты
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'html') node.innerHTML = value;
      else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
      else if (key === 'dataset') Object.assign(node.dataset, value);
      else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function store(key, value) {
    try {
      if (value === undefined) return window.localStorage.getItem(key);
      window.localStorage.setItem(key, value);
    } catch (e) { /* хранилище браузера может быть недоступно — не критично */ }
    return null;
  }

  const newId = () => (window.crypto && crypto.randomUUID ? crypto.randomUUID().replace(/-/g, '') : String(Date.now()) + Math.random().toString(16).slice(2));

  function debounce(fn, ms) {
    let t = null;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  function toast(text, ms = 2600) {
    const node = $('#toast');
    node.textContent = text;
    node.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { node.hidden = true; }, ms);
  }

  async function api(path, { method = 'GET', body } = {}) {
    const response = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail || detail; } catch (e) { /* пусто */ }
      throw new Error(detail);
    }
    return response.json();
  }

  const fmtDate = (iso) => {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  const PLOT_CONFIG = {
    responsive: true,
    locale: 'ru',
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'],
  };

  // ------------------------------------------------------------------ состояние
  const S = {
    sessionId: null,
    busy: false,
    health: null,
    liveUid: null,
    dashboards: new Map(),   // uid -> контроллер дашборда
    shareToken: null,
  };

  const EXAMPLES = [
    ['Дашборд', 'Собери дашборд по продажам: план-факт по регионам и торговым точкам за последний год'],
    ['Один график', 'Покажи план-факт продаж по регионам за последний квартал'],
    ['Сравнение', 'Сравни продажи по торговым точкам за год, выдели отстающих'],
    ['Скидки', 'Покажи по клиенту Сибирский Продукт за последние 3 года, какие скидки делались и почему'],
    ['Структура', 'Покажи структуру продаж по продуктам за последний месяц'],
    ['Сохранённое', 'Покажи мои дашборды'],
  ];

  // ================================================================== ЧАТ
  const chat = () => $('#chat');

  function scrollToBottom(node) {
    // мгновенно (без плавной прокрутки через всю историю) показываем начало нового ответа
    const box = chat();
    if (node && node.isConnected) {
      const top = node.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 12;
      box.scrollTo({ top: Math.max(top, 0), behavior: 'auto' });
    } else {
      box.scrollTo({ top: box.scrollHeight, behavior: 'auto' });
    }
  }

  function renderWelcome() {
    const h = S.health || {};
    const node = el('div', { class: 'welcome', id: 'welcome' },
      el('h1', { text: 'Спросите о продажах — получите живой дашборд' }),
      el('p', { class: 'lead', text: 'Бот сам находит данные в базе, собирает единую таблицу, считает то, что вы попросите словами, и строит интерактивный дашборд с фильтрами, переключателями и итоговыми цифрами. Дашборд можно сохранить и вернуться к нему позже.' }),
      el('div', { class: 'examples' }, EXAMPLES.map(([k, v]) => el('button', { class: 'example', type: 'button', onclick: () => send(v) },
        el('div', { class: 'k', text: k }), el('div', { class: 'v', text: v })))),
      el('div', { class: 'facts' },
        el('span', {}, 'База: ', el('b', { text: h.database === 'postgresql' ? 'PostgreSQL' : 'SQLite (встроенная)' })),
        el('span', {}, 'Данные: ', el('b', { text: h.data_from ? `${h.data_from} — ${h.data_to}` : '…' })),
        el('span', {}, 'Понимание запросов: ', el('b', { text: h.llm && h.llm !== 'выключена' ? `правила + LLM (${h.llm})` : 'встроенные правила' })),
      ));
    chat().append(node);
  }

  function removeWelcome() {
    const w = $('#welcome');
    if (w) w.remove();
  }

  function userMessage(text) {
    removeWelcome();
    const node = el('div', { class: 'msg user' }, el('div', { class: 'bubble-user', text }));
    chat().append(node);
    return node;
  }

  function botShell() {
    const avatar = el('div', { class: 'bot-avatar', html: '<svg viewBox="0 0 32 32" width="18" height="18"><rect x="7" y="17" width="4" height="8" rx="1.5" fill="#fff"/><rect x="14" y="11" width="4" height="14" rx="1.5" fill="#fff"/><rect x="21" y="7" width="4" height="18" rx="1.5" fill="#fff"/></svg>' });
    const body = el('div', { class: 'bot-body' });
    const node = el('div', { class: 'msg bot' }, el('div', { class: 'bot-row' }, avatar, body));
    return { node, body };
  }

  function typing() {
    const { node, body } = botShell();
    body.append(el('div', { class: 'typing' }, el('i'), el('i'), el('i')));
    chat().append(node);
    scrollToBottom();
    return node;
  }

  function renderBotMessage(message, { draft = null, fromHistory = false } = {}) {
    removeWelcome();
    const { node, body } = botShell();
    chat().append(node);
    for (const block of message.blocks || []) {
      const rendered = renderBlock(block, { draft, fromHistory });
      if (rendered) body.append(rendered);
    }
    if (!body.children.length) node.remove();
    if (!fromHistory) setChips(message.suggestions || []);
    // активируем графики после вставки в DOM (Plotly нужны размеры)
    requestAnimationFrame(() => {
      for (const ctrl of S.dashboards.values()) if (ctrl.pendingPlot) ctrl.plot();
    });
    return node;
  }

  function renderBlock(block, opts) {
    switch (block.type) {
      case 'text':
        return el('div', { class: 'text-block' + (block.tone === 'error' ? ' error' : ''), html: block.text });
      case 'sql':
        return el('details', { class: 'sql', open: block.open || null },
          el('summary', { text: `SQL-запрос к базе · ${Number(block.rows || 0).toLocaleString('ru-RU')} строк · только чтение` }),
          el('pre', { text: block.sql }));
      case 'table':
        return renderTable(block);
      case 'dashboard':
        if (block.stub) {
          if (opts.draft && opts.draft.uid === block.uid && opts.lastLive) return renderDashboard(opts.draft, { live: true });
          return renderStub(block);
        }
        return renderDashboard(block.dashboard, { live: true });
      case 'dashboard_update':
        if (!block.stub) updateLive(block.dashboard);
        return null;
      case 'dashboards':
        return renderList(block);
      case 'clarify':
        return renderClarify(block, opts);
      case 'download':
        return renderDownload(block, opts);
      case 'share':
        return renderShareLink(block);
      case 'history':
        return renderHistory(block);
      case 'highlight':
        if (!opts.fromHistory) highlight(block.target);
        return null;
      default:
        return null;
    }
  }

  function setChips(items) {
    const box = $('#chips');
    box.replaceChildren();
    for (const text of items.slice(0, 6)) {
      box.append(el('button', { class: 'chip', type: 'button', title: text, onclick: () => send(text) }, text));
    }
  }

  async function send(text) {
    text = (text || '').trim();
    if (!text || S.busy) return;
    const input = $('#input');
    input.value = '';
    autosize();
    userMessage(text);
    await roundtrip(() => api('/api/chat', { method: 'POST', body: { session_id: S.sessionId, text } }));
  }

  async function runAction(action, label) {
    if (S.busy) return;
    if (label) userMessage(label);
    await roundtrip(() => api('/api/action', { method: 'POST', body: { session_id: S.sessionId, action, label } }));
  }

  async function roundtrip(call) {
    S.busy = true;
    $('#send').disabled = true;
    const wait = typing();
    let node = null;
    try {
      const data = await call();
      wait.remove();
      node = renderBotMessage(data.message);
      refreshSide();
    } catch (err) {
      wait.remove();
      node = renderBotMessage({ blocks: [{ type: 'text', tone: 'error', text: 'Ошибка связи с сервером: ' + escapeHtml(err.message) }] });
    } finally {
      S.busy = false;
      $('#send').disabled = false;
      $('#input').focus();
    }
    if (node) scrollToBottom(node);
  }

  const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // ------------------------------------------------------------------ таблица
  function renderTable(block, existing) {
    const card = existing || el('div', { class: 'card table-card' });
    card.replaceChildren();
    const pages = Math.max(1, Math.ceil(block.total / block.page_size));
    const meta = block.preview
      ? `первые ${Math.min(block.page_size, block.total)} из ${block.total.toLocaleString('ru-RU')} строк`
      : `${block.total.toLocaleString('ru-RU')} строк`;
    const params = (extra = {}) => {
      const p = new URLSearchParams({ session_id: S.sessionId, page: 0, group_by: (block.group_by || []).join(','),
        sort_by: block.sort_by || '', sort_dir: block.sort_dir || 'desc', ...extra });
      if (block.top_n) p.set('top_n', block.top_n);
      return p;
    };
    const reload = async (extra) => {
      try {
        const data = await api('/api/table?' + params(extra).toString());
        renderTable(data, card);
      } catch (err) { toast(err.message); }
    };
    const groupSelect = el('select', { class: 'mini-select', 'aria-label': 'Группировка', onchange: (e) => reload({ group_by: e.target.value, page: 0 }) },
      el('option', { value: '' }, 'без группировки'),
      [['region', 'по регионам'], ['outlet', 'по точкам'], ['category', 'по категориям'], ['product', 'по продуктам'], ['period', 'по периодам'], ['client', 'по клиентам'], ['reason', 'по причинам']]
        .map(([v, l]) => el('option', { value: v, selected: (block.group_by || [])[0] === v }, l)));
    card.append(el('div', { class: 'table-head' },
      el('div', {}, el('div', { class: 'table-title', text: block.title }), el('div', { class: 'table-meta', text: meta + (Object.keys(block.filters || {}).length ? ' · с учётом фильтров' : '') })),
      el('div', { class: 'table-tools' },
        groupSelect,
        block.preview ? el('button', { class: 'btn btn-sm', type: 'button', onclick: () => reload({ page: 0, preview: 0 }) }, 'Показать всю') : null,
        el('a', { class: 'btn btn-sm', href: `/api/export?session_id=${encodeURIComponent(S.sessionId)}&format=csv`, download: '' }, 'CSV'),
        el('a', { class: 'btn btn-sm', href: `/api/export?session_id=${encodeURIComponent(S.sessionId)}&format=xlsx`, download: '' }, 'Excel'),
      )));
    const thead = el('tr', {}, block.columns.map((col) => {
      const sorted = block.sort_by === col.key;
      const th = el('th', {
        class: [col.numeric ? 'num' : '', col.calc ? 'calc' : '', col.highlight ? 'hl' : ''].join(' '),
        title: col.calc ? 'Вычисленная колонка — нажмите, чтобы отсортировать' : 'Нажмите, чтобы отсортировать',
        onclick: () => reload({ sort_by: col.key, sort_dir: sorted && block.sort_dir === 'desc' ? 'asc' : 'desc', page: 0 }),
      }, col.calc ? 'ƒ ' : '', col.label, sorted ? el('span', { class: 'sort-ind', text: block.sort_dir === 'asc' ? '▲' : '▼' }) : null);
      return th;
    }));
    const rows = block.rows.map((row) => el('tr', {}, row.map((cell, i) => {
      const col = block.columns[i];
      const long = !col.numeric && String(cell).length > 42;
      return el('td', { class: [col.numeric ? 'num' : '', col.calc ? 'calc' : '', col.highlight ? 'hl' : '', long ? 'trunc' : ''].join(' '), title: long ? cell : null, text: cell });
    })));
    card.append(el('div', { class: 'table-wrap' }, el('table', { class: 'data' }, el('thead', {}, thead), el('tbody', {}, rows))));
    if (!rows.length) card.append(el('div', { class: 'table-meta', text: 'Нет строк для выбранных фильтров.' }));
    if (!block.preview && pages > 1) {
      card.append(el('div', { class: 'pager' },
        el('button', { class: 'btn btn-sm', type: 'button', disabled: block.page <= 0, onclick: () => reload({ page: block.page - 1 }) }, '← Назад'),
        el('span', { text: `стр. ${block.page + 1} из ${pages}` }),
        el('button', { class: 'btn btn-sm', type: 'button', disabled: block.page >= pages - 1, onclick: () => reload({ page: block.page + 1 }) }, 'Вперёд →')));
    }
    if ((block.calcs || []).length) {
      card.append(el('div', { class: 'calc-note' }, block.calcs.map((c) => el('span', { class: 'calc-chip', style: 'padding-right:10px', title: c.formula }, 'ƒ ' + c.label))));
    }
    return card;
  }

  // ------------------------------------------------------------------ дашборд
  function renderStub(block) {
    return el('div', { class: 'stub' },
      el('span', { text: `Дашборд «${block.name || 'без названия'}» · ${block.period_label || ''} — предыдущее состояние (свёрнуто)` }));
  }

  function renderDashboard(payload, { live = true, share = false } = {}) {
    const ctrl = new Dashboard(payload, live ? 'live' : (share ? 'share' : 'snapshot'));
    if (live) {
      for (const other of liveControllers()) if (other !== ctrl) other.collapse();
      S.liveUid = payload.uid;
      S.liveCtrl = ctrl;
      $('#topTitle').textContent = payload.name || 'Дашборд';
    }
    return ctrl.node;
  }

  function liveControllers() {
    return Array.from(S.dashboards.values()).filter((c) => c.mode === 'live' && !c.collapsed);
  }

  function updateLive(payload) {
    const ctrl = S.liveCtrl;
    if (ctrl && !ctrl.collapsed) {
      ctrl.update(payload);
      ctrl.node.classList.remove('flash');
      void ctrl.node.offsetWidth;
      ctrl.node.classList.add('flash');
      $('#topTitle').textContent = payload.name || 'Дашборд';
    }
  }

  function highlight(target) {
    const ctrl = S.liveCtrl;
    if (!ctrl) return;
    const map = { filter: '.filters', cascade: '.filters', kpi: '.kpis', series: '.switches', grain: '.switches', switch: '.switches' };
    const node = $(map[target] || '.controls', ctrl.node);
    if (node) {
      node.classList.remove('flash');
      void node.offsetWidth;
      node.classList.add('flash');
      node.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  class Dashboard {
    constructor(payload, mode) {
      this.mode = mode;
      this.payload = payload;
      this.collapsed = false;
      this.tsList = [];
      this.plots = [];
      this.showTable = new Set();
      this.node = el('div', { class: 'card dash' });
      this.id = newId();
      S.dashboards.set(this.id, this);
      this.build();
    }

    dispose() {
      for (const ts of this.tsList) { try { ts.destroy(); } catch (e) { /* ок */ } }
      this.tsList = [];
      for (const div of this.plots) { try { Plotly.purge(div); } catch (e) { /* ок */ } }
      this.plots = [];
    }

    collapse() {
      if (this.collapsed) return;
      this.dispose();
      this.collapsed = true;
      const p = this.payload;
      const stub = el('div', { class: 'stub' },
        el('span', { text: `Дашборд «${p.name || 'без названия'}» · ${p.period_label} — предыдущее состояние` }),
        el('button', { class: 'btn btn-sm', type: 'button', onclick: () => this.expandSnapshot(stub) }, 'Показать снимок'));
      this.node.replaceWith(stub);
      this.node = stub;
      S.dashboards.delete(this.id);
    }

    expandSnapshot(stub) {
      const snap = new Dashboard(this.payload, 'snapshot');
      stub.replaceWith(snap.node);
      requestAnimationFrame(() => snap.plot());
    }

    update(payload) {
      this.payload = payload;
      this.build();
      this.plot();
    }

    async apply({ filters, switches }) {
      this.node.classList.add('loading');
      try {
        let data;
        if (this.mode === 'share') {
          data = await api(`/api/share/${S.shareToken}/render`, { method: 'POST', body: { filters: filters ?? this.payload.active_filters, switches: switches ?? this.currentSwitches() } });
        } else {
          data = await api('/api/draft/update', { method: 'POST', body: { session_id: S.sessionId, uid: this.payload.uid, filters, switches } });
        }
        this.update(data.dashboard);
      } catch (err) {
        toast(err.message);
      } finally {
        this.node.classList.remove('loading');
      }
    }

    currentSwitches() {
      const out = {};
      for (const sw of this.payload.switches || []) out[sw.key] = sw.value;
      return out;
    }

    collectFilters() {
      const out = {};
      for (const ts of this.tsList) {
        const values = ts.getValue();
        if (values && values.length) out[ts.__key] = Array.isArray(values) ? values : [values];
      }
      return out;
    }

    build() {
      this.dispose();
      const p = this.payload;
      const n = this.node;
      n.replaceChildren();
      const readOnly = this.mode !== 'live';

      // шапка
      const saved = p.saved ? el('span', { class: 'pill saved', text: `сохранён · версия ${p.saved.version}` }) : (readOnly ? null : el('span', { class: 'pill draft', text: 'черновик' }));
      const actions = el('div', { class: 'dash-actions' });
      if (this.mode === 'live') {
        actions.append(
          el('button', { class: 'btn btn-primary', type: 'button', onclick: () => runAction({ type: 'save' }, 'Сохрани дашборд') }, p.saved ? 'Сохранить изменения' : 'Сохранить'),
          el('button', { class: 'btn', type: 'button', onclick: (e) => this.menu(e, 'saveas') }, 'Сохранить как…'),
          el('button', { class: 'btn', type: 'button', onclick: () => runAction({ type: 'show_table' }, 'Покажи таблицу') }, 'Таблица'),
          el('button', { class: 'btn', type: 'button', onclick: (e) => this.menu(e, 'download') }, 'Скачать ▾'),
          el('button', { class: 'btn', type: 'button', onclick: () => runAction({ type: 'share' }, 'Поделись ссылкой') }, 'Поделиться'),
        );
      } else {
        actions.append(el('button', { class: 'btn', type: 'button', onclick: () => exportPng(this) }, 'Скачать PNG'));
      }
      n.append(el('div', { class: 'dash-head' },
        el('div', {},
          el('div', { class: 'dash-title', text: p.name || 'Дашборд' }),
          el('div', { class: 'dash-sub' },
            el('span', { class: 'pill', text: p.dataset_label }),
            el('span', { class: 'pill', text: `${p.period.label} · ${p.period_label}` }),
            saved,
            el('span', { text: `${Number(p.rows_filtered).toLocaleString('ru-RU')} из ${Number(p.rows_total).toLocaleString('ru-RU')} строк` }))),
        actions));

      const body = el('div', { class: 'dash-body' });
      n.append(body);

      // KPI
      body.append(el('div', { class: 'kpis' }, (p.kpis || []).map((k) => {
        const m = /^([−+-]?[\d\u00a0\s,.]+%?)(.*)$/.exec(k.display || '');
        const value = m && m[1].trim() ? el('div', { class: 'kpi-value' }, k.arrow ? el('span', { class: 'arrow', text: k.arrow }) : null, m[1].trim(),
          m[2].trim() ? el('span', { class: 'unit', text: m[2].trim() }) : null)
          : el('div', { class: 'kpi-value textual', text: k.display });
        return el('div', { class: `kpi ${k.status}`, title: k.full || k.display || '' },
          el('div', { class: 'kpi-label', text: k.label }), value, el('div', { class: 'kpi-sub', text: k.sub || '' }));
      })));

      // фильтры и переключатели
      const controls = el('div', { class: 'controls' });
      const filtersBox = el('div', { class: 'filters' });
      controls.append(filtersBox);
      const applyFilters = debounce(() => this.apply({ filters: this.collectFilters() }), 350);
      const byKey = {};
      for (const f of p.filters || []) {
        const select = el('select', { multiple: true, 'aria-label': f.label, placeholder: f.placeholder });
        for (const o of f.options) select.append(el('option', { value: o.value, selected: (f.selected || []).includes(o.value) }, o.label));
        const wrap = el('div', { class: 'filter' }, el('label', { text: f.label }), select);
        filtersBox.append(wrap);
        byKey[f.key] = { f, select };
      }
      const switchesBox = el('div', { class: 'switches' });
      for (const sw of p.switches || []) {
        const seg = el('div', { class: 'seg', role: 'group', 'aria-label': sw.label });
        for (const o of sw.options) {
          seg.append(el('button', { type: 'button', class: o.value === sw.value ? 'on' : '', 'aria-pressed': o.value === sw.value ? 'true' : 'false',
            onclick: () => { if (o.value !== sw.value) this.apply({ switches: { [sw.key]: o.value } }); } }, o.label));
        }
        switchesBox.append(el('div', { class: 'switch' }, el('span', { class: 'switch-label', text: sw.label }), seg));
      }
      controls.append(switchesBox);
      const calcs = p.calcs || [];
      const foot = el('div', { class: 'controls-foot' },
        el('div', { class: 'calc-chips' }, calcs.length ? 'Вычисления:' : 'Вычислений пока нет — попросите в чате, например «посчитай процент выполнения плана».',
          calcs.map((c) => el('span', { class: 'calc-chip', title: c.formula }, 'ƒ ' + c.label,
            this.mode === 'live' ? el('button', { type: 'button', title: 'Убрать колонку', 'aria-label': 'Убрать колонку ' + c.label, onclick: () => runAction({ type: 'remove_calc', ref: c.id }, `Убери колонку «${c.label}»`) }, '✕') : null))),
        Object.keys(p.active_filters || {}).length ? el('button', { class: 'link-btn', type: 'button', onclick: () => this.apply({ filters: {} }) }, 'Сбросить фильтры') : null);
      controls.append(foot);
      body.append(controls);

      // Tom Select для фильтров (поиск, мультивыбор, каскад)
      const cascade = (childKey) => {
        const child = byKey[childKey];
        if (!child || !child.f.parent || !child.ts) return;
        const parent = byKey[child.f.parent];
        const chosen = parent && parent.ts ? parent.ts.getValue() : [];
        const allowed = child.f.options.filter((o) => !chosen.length || chosen.includes(child.f.parent_map[o.value]));
        const keep = child.ts.getValue().filter((v) => allowed.some((o) => o.value === v));
        child.ts.clearOptions();
        child.ts.addOptions(allowed.map((o) => ({ value: o.value, text: o.label })));
        child.ts.setValue(keep, true);
        child.ts.settings.placeholder = chosen.length ? `Все (${allowed.length})` : child.f.placeholder;
        child.ts.inputState && child.ts.inputState();
      };
      for (const [key, item] of Object.entries(byKey)) {
        const ts = new TomSelect(item.select, {
          plugins: ['remove_button'],
          maxOptions: 2000,
          hidePlaceholder: true,
          hideSelected: true,
          closeAfterSelect: false,
          placeholder: item.f.placeholder,
          searchField: ['text'],
          render: { no_results: () => '<div class="no-results">Ничего не найдено</div>' },
          onChange: () => {
            for (const other of Object.values(byKey)) if (other.f.parent === key) cascade(other.f.key);
            applyFilters();
          },
        });
        ts.__key = key;
        item.ts = ts;
        this.tsList.push(ts);
      }
      for (const item of Object.values(byKey)) if (item.f.parent) cascade(item.f.key);

      // графики
      const charts = el('div', { class: 'charts' + ((p.charts || []).length === 1 ? ' one' : '') });
      this.chartSlots = [];
      for (const c of p.charts || []) {
        const plot = el('div', { class: 'chart', role: 'img', 'aria-label': c.title || 'График' });
        const tableBox = el('div', { class: 'chart-table', hidden: !this.showTable.has(c.id) });
        if (this.showTable.has(c.id) && c.table) tableBox.append(miniTable(c.table));
        const tools = el('div', { class: 'chart-tools' },
          el('span', { class: 'pill', title: 'Номер графика — на него можно ссылаться в чате: «перестрой третий график»', text: `График ${c.index}` }),
          el('span', { style: 'flex:1' }),
          c.table ? el('button', { class: 'btn btn-sm btn-ghost', type: 'button', onclick: () => {
            if (this.showTable.has(c.id)) { this.showTable.delete(c.id); tableBox.hidden = true; tableBox.replaceChildren(); }
            else { this.showTable.add(c.id); tableBox.hidden = false; tableBox.replaceChildren(miniTable(c.table)); }
          } }, 'Данные') : null,
          el('button', { class: 'btn btn-sm btn-ghost', type: 'button', title: 'Развернуть на всю ширину', 'aria-label': 'Развернуть график', onclick: (e) => {
            const card = e.currentTarget.closest('.chart-card');
            card.classList.toggle('wide');
            this.wide = this.wide || new Set();
            if (card.classList.contains('wide')) this.wide.add(c.id); else this.wide.delete(c.id);
            setTimeout(() => { try { Plotly.Plots.resize(plot); } catch (err) { /* ок */ } }, 30);
          } }, '⤢'),
          el('button', { class: 'btn btn-sm btn-ghost', type: 'button', onclick: () => Plotly.downloadImage(plot, { format: 'png', filename: (c.title || 'график').slice(0, 60), scale: 2, width: 1100, height: 520 }) }, 'PNG'));
        charts.append(el('div', { class: 'chart-card' + (this.wide && this.wide.has(c.id) ? ' wide' : ''), dataset: { chart: c.index } }, tools, plot, tableBox));
        this.chartSlots.push({ div: plot, chart: c });
      }
      body.append(charts);

      // резюме
      if ((p.summary || []).length) {
        body.append(el('div', { class: 'summary' }, el('div', { class: 'summary-title', text: 'Главное' }),
          el('ul', {}, p.summary.map((s) => el('li', { html: s })))));
      }
      if (this.mode === 'live' && (p.history || []).length) {
        body.append(el('details', { class: 'hist' }, el('summary', { text: `История изменений (${p.history.length})` }),
          el('ol', {}, p.history.map((h) => el('li', {}, el('span', { class: 'table-meta', text: fmtDate(h.ts) + ' — ' }), h.text)))));
      }
      this.pendingPlot = true;
    }

    plot() {
      this.pendingPlot = false;
      if (!this.node.isConnected) return;
      const shared = [];
      for (const slot of this.chartSlots || []) {
        const fig = slot.chart.figure;
        const layout = Object.assign({}, fig.layout, { autosize: true });
        delete layout.height;
        Plotly.newPlot(slot.div, fig.data, layout, Object.assign({}, PLOT_CONFIG, {
          toImageButtonOptions: { format: 'png', filename: (slot.chart.title || 'график').slice(0, 60), scale: 2 },
        }));
        this.plots.push(slot.div);
        if (slot.chart.x_shared) shared.push(slot.div);
      }
      // синхронный зум графиков с общей осью периодов
      if (shared.length > 1) {
        let syncing = false;
        for (const div of shared) {
          div.on('plotly_relayout', (ev) => {
            if (syncing) return;
            const update = {};
            if (ev['xaxis.range[0]'] !== undefined) {
              update['xaxis.range'] = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
            } else if (ev['xaxis.autorange']) {
              update['xaxis.autorange'] = true;
            } else if (ev['xaxis.range']) {
              update['xaxis.range'] = ev['xaxis.range'];
            } else return;
            syncing = true;
            Promise.all(shared.filter((d) => d !== div).map((d) => Plotly.relayout(d, update))).finally(() => { syncing = false; });
          });
        }
      }
    }

    menu(event, kind) {
      event.stopPropagation();
      const anchor = event.currentTarget.parentElement;
      $$('.menu').forEach((m) => m.remove());
      const box = el('div', { class: 'menu' });
      if (kind === 'download') {
        const go = (format, label) => { box.remove(); runAction({ type: 'export', format }, label); };
        box.append(
          el('button', { type: 'button', onclick: () => go('html', 'Скачай дашборд в HTML') }, 'HTML-страница', el('small', { text: 'интерактивные графики для отчёта' })),
          el('button', { type: 'button', onclick: () => { box.remove(); exportPng(this); } }, 'Картинка PNG', el('small', { text: 'весь дашборд одним изображением' })),
          el('button', { type: 'button', onclick: () => go('xlsx', 'Выгрузи таблицу в Excel') }, 'Excel', el('small', { text: 'таблица с вычислениями и формулами' })),
          el('button', { type: 'button', onclick: () => go('csv', 'Выгрузи таблицу в CSV') }, 'CSV', el('small', { text: 'разделитель «;»' })),
        );
      } else {
        const input = el('input', { class: 'input', type: 'text', placeholder: 'Название дашборда', value: this.payload.name || '' });
        const submit = () => {
          const name = input.value.trim();
          if (!name) return;
          box.remove();
          runAction({ type: 'save', name, as_new: true }, `Сохрани как «${name}»`);
        };
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') submit(); if (e.key === 'Escape') box.remove(); });
        box.append(el('div', { style: 'padding:6px;display:flex;flex-direction:column;gap:6px' },
          el('small', { style: 'color:var(--muted)', text: 'Сохранить как новый дашборд:' }), input,
          el('button', { class: 'btn btn-primary', type: 'button', onclick: submit, style: 'justify-content:center' }, 'Сохранить')));
        setTimeout(() => input.select(), 0);
      }
      anchor.append(box);
      const close = (e) => { if (!box.contains(e.target)) { box.remove(); document.removeEventListener('click', close); } };
      setTimeout(() => document.addEventListener('click', close), 0);
    }
  }

  function miniTable(table) {
    const head = el('tr', {}, table.columns.map((c, i) => el('th', { class: i ? 'num' : '', text: c })));
    const rows = table.rows.map((r) => el('tr', {}, r.map((v, i) => el('td', { class: i ? 'num' : '', text: v }))));
    return el('div', { class: 'table-wrap' }, el('table', { class: 'data' }, el('thead', {}, head), el('tbody', {}, rows)),
      table.truncated ? el('div', { class: 'table-meta', style: 'padding:6px', text: 'Показаны первые строки' }) : null);
  }

  // ------------------------------------------------------------------ PNG всего дашборда
  async function exportPng(ctrl) {
    if (!ctrl || ctrl.collapsed) { toast('Дашборд не найден'); return; }
    toast('Собираю картинку…');
    const p = ctrl.payload;
    const W = 1600;
    const S2 = 2;
    const pad = 32;
    const colW = (W - pad * 3) / 2;
    const chartH = 440;
    const images = [];
    for (const slot of ctrl.chartSlots) {
      const url = await Plotly.toImage(slot.div, { format: 'png', width: Math.round(colW), height: chartH, scale: S2 });
      images.push(await new Promise((resolve) => { const img = new Image(); img.onload = () => resolve(img); img.src = url; }));
    }
    const cols = images.length === 1 ? 1 : 2;
    const rows = Math.ceil(images.length / cols);
    const summaryLines = (p.summary || []).map((s) => '• ' + s.replace(/<[^>]+>/g, ''));
    const H = 110 + 110 + rows * (chartH + 16) + 40 + summaryLines.length * 24 + 40;
    const canvas = document.createElement('canvas');
    canvas.width = W * S2;
    canvas.height = H * S2;
    const g = canvas.getContext('2d');
    g.scale(S2, S2);
    g.fillStyle = '#f3f3f0'; g.fillRect(0, 0, W, H);
    g.fillStyle = '#0b0b0b'; g.font = '700 28px system-ui, Segoe UI, sans-serif';
    g.fillText(p.name || 'Дашборд', pad, 52);
    g.fillStyle = '#52514e'; g.font = '15px system-ui, Segoe UI, sans-serif';
    const flabels = Object.fromEntries((p.filters || []).map((f) => [f.key, f.label]));
    const flt = Object.entries(p.active_filters || {}).map(([k, v]) => `${flabels[k] || k}: ${v.slice(0, 3).join(', ')}${v.length > 3 ? '…' : ''}`).join('; ') || 'без фильтров';
    g.fillText(`${p.dataset_label} · ${p.period.label} (${p.period_label}) · ${flt}`, pad, 80);
    const kpis = p.kpis || [];
    const kw = (W - pad * 2 - (kpis.length - 1) * 12) / Math.max(kpis.length, 1);
    kpis.forEach((k, i) => {
      const x = pad + i * (kw + 12);
      const y = 104;
      g.fillStyle = '#ffffff'; roundRect(g, x, y, kw, 92, 12); g.fill();
      g.strokeStyle = '#e1e0d9'; g.stroke();
      if (k.status !== 'neutral') { g.fillStyle = k.status === 'good' ? '#0ca30c' : '#d03b3b'; g.fillRect(x + 12, y, kw - 24, 3); }
      g.fillStyle = '#52514e'; g.font = '13px system-ui, Segoe UI, sans-serif'; g.fillText(fit(g, k.label, kw - 24), x + 12, y + 26);
      g.fillStyle = k.status === 'good' ? '#006300' : k.status === 'bad' ? '#c42f2f' : '#0b0b0b';
      g.font = '700 24px system-ui, Segoe UI, sans-serif'; g.fillText(fit(g, (k.arrow ? k.arrow + ' ' : '') + k.display, kw - 24), x + 12, y + 58);
      g.fillStyle = '#898781'; g.font = '12px system-ui, Segoe UI, sans-serif'; g.fillText(fit(g, k.sub || '', kw - 24), x + 12, y + 80);
    });
    let y0 = 220;
    images.forEach((img, i) => {
      const cw = cols === 1 ? W - pad * 2 : colW;
      const x = pad + (i % cols) * (colW + pad);
      const y = y0 + Math.floor(i / cols) * (chartH + 16);
      g.fillStyle = '#fcfcfb'; roundRect(g, x, y, cw, chartH, 12); g.fill(); g.strokeStyle = '#e1e0d9'; g.stroke();
      g.drawImage(img, x, y, cw, chartH);
    });
    let y = y0 + rows * (chartH + 16) + 24;
    g.fillStyle = '#0b0b0b'; g.font = '700 16px system-ui, Segoe UI, sans-serif'; g.fillText('Главное', pad, y);
    g.font = '14px system-ui, Segoe UI, sans-serif'; g.fillStyle = '#2b2b29';
    summaryLines.forEach((line) => { y += 24; g.fillText(fit(g, line, W - pad * 2), pad, y); });
    const link = document.createElement('a');
    link.download = ((p.name || 'дашборд').replace(/[^\wа-яё.-]+/gi, '_')).slice(0, 60) + '.png';
    link.href = canvas.toDataURL('image/png');
    document.body.append(link);
    link.click();
    link.remove();
    toast('Картинка сохранена');
  }

  function roundRect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
  }

  function fit(g, text, width) {
    text = String(text);
    if (g.measureText(text).width <= width) return text;
    while (text.length > 1 && g.measureText(text + '…').width > width) text = text.slice(0, -1);
    return text + '…';
  }

  // ------------------------------------------------------------------ список дашбордов
  function renderList(block) {
    const card = el('div', { class: 'card list-card' });
    const listBox = el('div', { class: 'dlist' });
    const search = el('input', { class: 'input', type: 'search', placeholder: 'Поиск по названию или запросу', value: block.query || '' });
    const sort = el('select', { class: 'mini-select', 'aria-label': 'Сортировка' },
      [['updated', 'по дате изменения'], ['created', 'по дате создания'], ['name', 'по названию']].map(([v, l]) => el('option', { value: v, selected: block.sort === v }, l)));
    const draw = (items) => {
      listBox.replaceChildren();
      if (!items.length) { listBox.append(el('div', { class: 'side-empty', text: 'Ничего не найдено.' })); return; }
      for (const item of items) listBox.append(listRow(item));
    };
    const reload = debounce(async () => {
      try {
        const data = await api(`/api/dashboards?q=${encodeURIComponent(search.value)}&sort=${sort.value}`);
        draw(data.items);
      } catch (err) { toast(err.message); }
    }, 250);
    search.addEventListener('input', reload);
    sort.addEventListener('change', reload);
    card.append(el('div', { class: 'list-tools' }, search, sort), listBox);
    draw(block.items || []);
    return card;
  }

  function listRow(item) {
    const row = el('div', { class: 'drow' });
    const info = el('div', {},
      el('div', { class: 'n', text: item.name }),
      el('div', { class: 'q', text: item.query ? `«${item.query}»` : item.description || '' }),
      el('div', { class: 'm', text: `создан ${fmtDate(item.created_at)} · изменён ${fmtDate(item.updated_at)} · версия ${item.version}` }));
    const acts = el('div', { class: 'acts' },
      el('button', { class: 'btn btn-sm btn-primary', type: 'button', onclick: () => runAction({ type: 'open', id: item.id }, `Открой дашборд «${item.name}»`) }, 'Открыть'),
      el('button', { class: 'btn btn-sm', type: 'button', onclick: () => {
        if ($('.rename-box', info)) return;
        const input = el('input', { class: 'input', type: 'text', value: item.name });
        const ok = () => { const v = input.value.trim(); if (v && v !== item.name) runAction({ type: 'rename', id: item.id, new_name: v }, `Переименуй «${item.name}» в «${v}»`); };
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') ok(); });
        info.append(el('div', { class: 'rename-box' }, input, el('button', { class: 'btn btn-sm btn-primary', type: 'button', onclick: ok }, 'OK')));
        input.focus(); input.select();
      } }, 'Переименовать'),
      el('button', { class: 'btn btn-sm', type: 'button', onclick: () => runAction({ type: 'copy', id: item.id }, `Сделай копию «${item.name}»`) }, 'Копия'),
      el('button', { class: 'btn btn-sm', type: 'button', onclick: () => runAction({ type: 'delete', id: item.id }, `Удали дашборд «${item.name}»`) }, 'Удалить'));
    row.append(info, acts);
    return row;
  }

  // ------------------------------------------------------------------ прочие блоки
  function renderClarify(block, opts) {
    const box = el('div', { class: 'clarify' });
    if (block.question) box.append(el('div', { class: 'q', html: block.question }));
    const options = el('div', { class: 'options' + (opts.fromHistory ? ' used' : '') });
    for (const option of block.options || []) {
      options.append(el('button', { class: 'btn btn-sm', type: 'button', onclick: () => { options.classList.add('used'); runAction(option.action, option.label); } }, option.label));
    }
    box.append(options);
    if (!(block.options || []).length) box.append(el('div', { class: 'table-meta', text: 'Напишите ответ в поле ввода.' }));
    return box;
  }

  function renderDownload(block, opts) {
    const ext = (block.format || '').toUpperCase();
    if (block.client_side) {
      if (!opts.fromHistory) setTimeout(() => exportPng(S.liveCtrl), 50);
      return el('div', { class: 'file' }, el('div', { class: 'ic', text: 'PNG' }), el('div', { class: 'fn', text: block.filename }),
        el('button', { class: 'btn btn-sm', type: 'button', onclick: () => exportPng(S.liveCtrl) }, 'Скачать ещё раз'));
    }
    const link = el('a', { class: 'btn btn-sm btn-primary', href: block.url, download: block.filename }, 'Скачать');
    if (!opts.fromHistory) setTimeout(() => link.click(), 80);
    return el('div', { class: 'file' }, el('div', { class: 'ic', text: ext }), el('div', { class: 'fn', text: block.filename }), link);
  }

  function renderShareLink(block) {
    const url = location.origin + block.url;
    const input = el('input', { class: 'input', type: 'text', value: url, readonly: true });
    return el('div', { class: 'file', style: 'max-width:760px' }, el('div', { class: 'ic', text: '🔗' }), input,
      el('button', { class: 'btn btn-sm', type: 'button', onclick: async () => {
        try { await navigator.clipboard.writeText(url); toast('Ссылка скопирована'); } catch (e) { input.select(); toast('Скопируйте ссылку вручную'); }
      } }, 'Копировать'),
      el('a', { class: 'btn btn-sm', href: block.url, target: '_blank', rel: 'noopener' }, 'Открыть'));
  }

  function renderHistory(block) {
    const card = el('div', { class: 'card hist-card' });
    card.append(el('h4', { text: `История изменений «${block.name || 'дашборда'}»` }));
    card.append(el('ol', {}, (block.items || []).map((h) => el('li', {}, el('span', { class: 'ts', text: fmtDate(h.ts) }), h.text))));
    if ((block.calcs || []).length) {
      card.append(el('h4', { text: 'Добавленные вычисления (можно убрать)' }));
      card.append(el('div', { class: 'calc-note' }, block.calcs.map((c) => el('span', { class: 'calc-chip', title: c.formula }, 'ƒ ' + c.label,
        el('button', { type: 'button', title: 'Убрать', onclick: () => runAction({ type: 'remove_calc', ref: c.id }, `Убери колонку «${c.label}»`) }, '✕')))));
    }
    if (block.saved) {
      card.append(el('h4', { text: 'Сохранённые версии', style: 'margin-top:10px' }));
      card.append(el('div', { class: 'options' }, (block.versions || []).map((v) => el('button', { class: 'btn btn-sm', type: 'button', title: v.note || '',
        onclick: () => runAction({ type: 'revert', version: v.version }, `Откати к версии ${v.version}`) }, `Версия ${v.version} · ${fmtDate(v.created_at)}`))));
    } else {
      card.append(el('div', { class: 'table-meta', text: 'Дашборд ещё не сохранён — версии появятся после сохранения. Отменить последний шаг: «Отмени».' }));
    }
    return card;
  }

  // ------------------------------------------------------------------ боковая панель
  async function refreshSide() {
    const q = $('#sideSearch').value;
    const sort = $('#sideSort').value;
    try {
      const data = await api(`/api/dashboards?q=${encodeURIComponent(q)}&sort=${sort}`);
      const box = $('#sideList');
      box.replaceChildren();
      if (!data.items.length) {
        box.append(el('div', { class: 'side-empty', text: q ? 'Ничего не найдено.' : 'Сохранённых дашбордов пока нет. Постройте дашборд и нажмите «Сохранить».' }));
        return;
      }
      const activeId = S.liveCtrl && S.liveCtrl.payload.saved ? S.liveCtrl.payload.saved.id : null;
      for (const item of data.items) {
        box.append(el('button', { class: 'side-item' + (item.id === activeId ? ' active' : ''), type: 'button', title: item.query || item.name,
          onclick: () => { $('#app').classList.remove('side-open'); runAction({ type: 'open', id: item.id }, `Открой дашборд «${item.name}»`); } },
        el('div', { class: 't', text: item.name }), el('div', { class: 'd', text: `${fmtDate(item.updated_at)} · ${item.query || item.description || ''}` })));
      }
    } catch (err) { /* список не критичен */ }
  }

  // ------------------------------------------------------------------ запуск
  function autosize() {
    const input = $('#input');
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  }

  async function initChat() {
    S.sessionId = store('bibot_session') || newId();
    store('bibot_session', S.sessionId);

    $('#form').addEventListener('submit', (e) => { e.preventDefault(); send($('#input').value); });
    $('#input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send($('#input').value); }
    });
    $('#input').addEventListener('input', autosize);
    $('#newChat').addEventListener('click', async () => {
      S.sessionId = newId();
      store('bibot_session', S.sessionId);
      for (const ctrl of S.dashboards.values()) ctrl.dispose();
      S.dashboards.clear();
      S.liveCtrl = null;
      chat().replaceChildren();
      $('#topTitle').textContent = 'Новый диалог';
      await api('/api/session/new', { method: 'POST', body: { session_id: S.sessionId } }).catch(() => null);
      renderWelcome();
      setChips(EXAMPLES.map((e) => e[1]).slice(0, 4));
      $('#app').classList.remove('side-open');
    });
    $('#helpBtn').addEventListener('click', () => send('Что ты умеешь?'));
    $('#toggleSide').addEventListener('click', () => $('#app').classList.toggle('side-open'));
    $('#sideSearch').addEventListener('input', debounce(refreshSide, 250));
    $('#sideSort').addEventListener('change', refreshSide);

    try {
      S.health = await api('/api/health');
      const h = S.health;
      $('#status').innerHTML = `База: <b>${h.database === 'postgresql' ? 'PostgreSQL' : 'SQLite'}</b> · только чтение<br>Данные: <b>${h.data_from} — ${h.data_to}</b><br>Понимание запросов: <b>${h.llm !== 'выключена' ? 'правила + ' + escapeHtml(h.llm) : 'встроенные правила'}</b>`;
    } catch (err) {
      $('#status').textContent = 'Нет связи с сервером: ' + err.message;
    }

    let session = null;
    try { session = await api(`/api/session?session_id=${encodeURIComponent(S.sessionId)}`); } catch (e) { session = null; }
    const messages = session ? session.messages : [];
    if (!messages.length) {
      renderWelcome();
      setChips(EXAMPLES.map((e) => e[1]).slice(0, 4));
    } else {
      const draft = session.draft;
      // индекс последнего упоминания текущего дашборда
      let lastIdx = -1;
      messages.forEach((m, i) => { if (m.role === 'bot' && draft && (m.blocks || []).some((b) => (b.type === 'dashboard' || b.type === 'dashboard_update') && b.uid === draft.uid)) lastIdx = i; });
      let lastBot = null;
      messages.forEach((m, i) => {
        if (m.role === 'user') { userMessage(m.text); return; }
        const blocks = (m.blocks || []).map((b) => {
          if (b.type === 'dashboard_update') return null;
          return b;
        }).filter(Boolean);
        const isLast = i === lastIdx;
        const { node, body } = botShell();
        chat().append(node);
        for (const b of blocks) {
          let rendered;
          if (b.type === 'dashboard' && b.stub && draft && isLast && b.uid === draft.uid) rendered = renderDashboard(draft, { live: true });
          else rendered = renderBlock(b, { fromHistory: true, draft });
          if (rendered) body.append(rendered);
        }
        if (isLast && draft && !blocks.some((b) => b.type === 'dashboard')) body.append(renderDashboard(draft, { live: true }));
        if (!body.children.length) node.remove();
        lastBot = m;
      });
      if (draft && lastIdx === -1) {
        const { node, body } = botShell();
        chat().append(node);
        body.append(renderDashboard(draft, { live: true }));
      }
      setChips(lastBot ? lastBot.suggestions || [] : []);
      requestAnimationFrame(() => {
        for (const ctrl of S.dashboards.values()) if (ctrl.pendingPlot) ctrl.plot();
        scrollToBottom();
        setTimeout(() => scrollToBottom(), 300);
      });
    }
    refreshSide();
    $('#input').focus();
  }

  async function initShare() {
    S.shareToken = decodeURIComponent(location.pathname.split('/share/')[1] || '').replace(/\/$/, '');
    $('#app').hidden = true;
    const shell = $('#shareShell');
    shell.hidden = false;
    try {
      const data = await api(`/api/share/${encodeURIComponent(S.shareToken)}`);
      document.title = data.name + ' — Puzzle BI-бот';
      shell.append(el('div', { class: 'share-top' },
        el('b', { text: 'Puzzle BI-бот' }), el('span', { text: `· дашборд по ссылке · версия ${data.version} от ${fmtDate(data.updated_at)} · только просмотр` })));
      const ctrl = new Dashboard(data.dashboard, 'share');
      shell.append(ctrl.node);
      requestAnimationFrame(() => ctrl.plot());
    } catch (err) {
      shell.append(el('div', { class: 'text-block error', text: 'Не удалось открыть дашборд: ' + err.message }));
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (location.pathname.startsWith('/share/')) initShare();
    else initChat();
  });
})();
