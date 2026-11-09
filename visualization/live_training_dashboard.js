/* Local file dashboard. Only reads a generated script and saved image paths. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const fmt = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : '—';
  const number = value => Number.isFinite(value) ? value.toLocaleString('en-US') : '—';
  let data = null, galleryName = null, loading = false;

  function svgElement(name, attrs = {}, text) {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function draw() {
    if (!data) return;
    const svg = $('plot');
    svg.replaceChildren();
    const kind = $('lossKind').value, xKey = $('xKind').value;
    const groups = $('range').value === 'all' ? [data.history, data.current] : [data.current];
    const points = groups.flat().filter(p => p[kind] && Number.isFinite(p[kind].train) && Number.isFinite(p[kind].dev));
    if (!points.length) {
      svg.append(svgElement('text', { x: 540, y: 170, 'text-anchor': 'middle', fill: '#64748b' }, '等待首个完整固定监测结果'));
      return;
    }
    const W = 1080, H = 350, left = 70, right = 22, top = 26, bottom = 48;
    let xmin = Math.min(...points.map(p => p[xKey])), xmax = Math.max(...points.map(p => p[xKey]));
    if (xmax === xmin) { xmin = Math.max(0, xmin - 500); xmax += 500; }
    const values = points.flatMap(p => [p[kind].train, p[kind].dev]);
    const lo = Math.min(...values), hi = Math.max(...values), span = Math.max(hi - lo, .02);
    const ymin = $('zero').checked ? 0 : Math.max(0, lo - span * .12);
    const ymax = hi + span * .14;
    const x = value => left + (value - xmin) / (xmax - xmin) * (W - left - right);
    const y = value => H - bottom - (value - ymin) / (ymax - ymin) * (H - top - bottom);
    for (let i = 0; i <= 5; i++) {
      const value = ymin + (ymax - ymin) * i / 5;
      svg.append(svgElement('line', { x1: left, x2: W - right, y1: y(value), y2: y(value), stroke: '#e2e8f0' }));
      svg.append(svgElement('text', { x: left - 10, y: y(value) + 4, 'text-anchor': 'end', fill: '#64748b', 'font-size': 12 }, value.toFixed(3)));
      const xv = xmin + (xmax - xmin) * i / 5;
      svg.append(svgElement('text', { x: x(xv), y: H - bottom + 23, 'text-anchor': 'middle', fill: '#64748b', 'font-size': 12 }, Math.round(xv).toLocaleString('en-US')));
    }
    svg.append(svgElement('text', { x: left, y: 15, fill: '#64748b', 'font-size': 12 }, $('lossKind').selectedOptions[0].textContent));
    svg.append(svgElement('text', { x: W - right, y: H - 5, 'text-anchor': 'end', fill: '#64748b', 'font-size': 12 }, $('xKind').selectedOptions[0].textContent));
    if ($('range').value === 'all' && data.current.length) {
      const switchX = x(data.current[0][xKey]);
      svg.append(svgElement('line', { x1: switchX, x2: switchX, y1: top, y2: H - bottom, stroke: '#94a3b8', 'stroke-dasharray': '3 4' }));
      svg.append(svgElement('text', { x: Math.min(switchX + 6, W - 165), y: top + 12, fill: '#64748b', 'font-size': 12 }, 'BF16 / batch4迁移起点'));
    }
    groups.forEach(group => {
      const good = group.filter(p => p[kind] && Number.isFinite(p[kind].train) && Number.isFinite(p[kind].dev));
      if (!good.length) return;
      const old = good[0].stage.startsWith('FP32');
      for (const [field, color] of [['train', '#2563eb'], ['dev', '#ea580c']]) {
        svg.append(svgElement('polyline', { points: good.map(p => `${x(p[xKey])},${y(p[kind][field])}`).join(' '),
          fill: 'none', stroke: color, 'stroke-width': old ? 2 : 2.8, 'stroke-dasharray': old ? '6 4' : '', opacity: old ? .5 : 1 }));
        good.forEach(p => svg.append(svgElement('circle', { cx: x(p[xKey]), cy: y(p[kind][field]), r: old ? 3 : 4, fill: color, opacity: old ? .5 : 1 })));
      }
    });
    const guide = svgElement('line', { y1: top, y2: H - bottom, stroke: '#64748b', 'stroke-dasharray': '3 3', visibility: 'hidden' });
    svg.append(guide);
    svg.onpointermove = event => {
      const box = svg.getBoundingClientRect();
      const mx = (event.clientX - box.left) / box.width * W;
      const nearest = points.reduce((a, b) => Math.abs(x(a[xKey]) - mx) <= Math.abs(x(b[xKey]) - mx) ? a : b);
      guide.setAttribute('x1', x(nearest[xKey])); guide.setAttribute('x2', x(nearest[xKey])); guide.setAttribute('visibility', 'visible');
      $('hover').textContent = `${nearest.stage} · 更新 ${number(nearest.steps)} · 样本次 ${number(nearest.seen)} · train500 ${fmt(nearest[kind].train)} · dev500 ${fmt(nearest[kind].dev)}` +
        (nearest.event === 'precision_migration_baseline' ? ' · 迁移基线，无参数更新' : '');
    };
    svg.onpointerleave = () => guide.setAttribute('visibility', 'hidden');
    const latest = data.current.at(-1);
    $('chartNote').textContent = `${data.data_note} 本次已有 ${data.current.length} 个监测点；最近完整监测位于累计第 ${number(latest?.steps)} 次更新。` +
      ($('range').value === 'all' ? ' 虚线是旧FP32，实线是本次BF16；不跨精度拼接。' : '');
  }

  function renderImages(gallery) {
    if (!gallery) return;
    $('imageStep').textContent = `图片来自第 ${number(gallery.steps)} 次更新`;
    $('imageNote').textContent = `${gallery.label}。图片在新Loss-best完整发布后自动切换；曲线与当前训练步数独立刷新。`;
    if (galleryName === gallery.name) return;
    galleryName = gallery.name;
    const fragment = document.createDocumentFragment();
    gallery.cases.forEach((row, i) => {
      const article = document.createElement('article'); article.className = 'image-card';
      const title = document.createElement('h3'); title.textContent = `${String(i + 1).padStart(3, '0')} · ${row.member[0]} : ${row.member[1]}`; article.append(title);
      const pair = document.createElement('div'); pair.className = 'pair';
      [['Reference', row.reference], ['生成图片', row.generated]].forEach(([label, source]) => {
        const figure = document.createElement('figure'), caption = document.createElement('figcaption'), anchor = document.createElement('a'), image = document.createElement('img');
        caption.textContent = label; anchor.href = source; image.src = source; image.loading = 'lazy'; image.alt = label;
        anchor.append(image); figure.append(caption, anchor); pair.append(figure);
      });
      article.append(pair); fragment.append(article);
    });
    $('images').replaceChildren(fragment);
  }

  function update() {
    data = window.E102_DASHBOARD;
    if (!data) return;
    const last = data.current.at(-1), heartbeat = data.heartbeat || {}, state = data.state || {};
    $('step').textContent = number(heartbeat.steps ?? state.steps);
    $('seen').textContent = `累计 ${number(heartbeat.seen_count ?? state.seen_count)} 样本次 · BF16 / batch4`;
    $('trainValue').textContent = fmt(last?.total.train); $('devValue').textContent = fmt(last?.total.dev);
    $('trainStep').textContent = `第 ${number(last?.steps)} 次更新的固定监测`;
    $('batchValue').textContent = fmt(heartbeat.loss);
    const stopped = ['user_stopped', 'time_limit', 'early_stopped', 'epoch_limit', 'failed'].includes(state.status);
    $('runStatus').textContent = stopped ? `训练已结束 · ${state.status}` : state.phase === 'monitor' ? '训练运行中 · 正在固定监测' : '训练运行中';
    $('updated').textContent = `页面数据更新于 ${new Date(data.generated_at).toLocaleTimeString('zh-CN', { hour12: false })}`;
    draw(); renderImages(data.gallery);
  }

  function refresh() {
    if (loading) return;
    loading = true;
    const script = document.createElement('script');
    script.src = `dashboard_data.js?refresh=${Date.now()}`;
    script.onload = () => { loading = false; script.remove(); };
    script.onerror = () => { loading = false; script.remove(); $('updated').textContent = '暂时无法读取更新，保留已有数据，稍后自动重试'; };
    document.body.append(script);
  }
  window.addEventListener('e102-data', update);
  ['range', 'lossKind', 'xKind', 'zero'].forEach(id => $(id).addEventListener('change', draw));
  $('refresh').addEventListener('click', refresh);
  refresh(); setInterval(refresh, 10000);
})();
