(() => {
  'use strict';
  const D = window.NTL_DATA;
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  if (!D || !D.rows?.length) { $('main').textContent = 'The dashboard data did not load. Please reload this page.'; return; }
  const options = (select, values) => values.forEach(value => { const n = el('option', value); n.value = value; select.append(n); });
  const models = Object.keys(D.model_counts).sort();
  const prompts = [...new Set(D.rows.map(r => r.prompt_id))].sort();
  const promptRows = new Map(prompts.map(p => [p, D.rows.filter(r => r.prompt_id === p)]));
  const stamp = new Date(D.checked_at).toLocaleString('en-GB', {timeZone:'UTC',dateStyle:'medium',timeStyle:'short'}) + ' UTC';
  const pages = [...document.querySelectorAll('.page')];
  function navigate(id, scroll = true) {
    if (!pages.some(p => p.id === id)) id = 'start';
    pages.forEach(p => { p.hidden = p.id !== id; });
    document.querySelectorAll('[data-page]').forEach(b => { if (b.dataset.page === id) b.setAttribute('aria-current','page'); else b.removeAttribute('aria-current'); });
    if (location.hash !== '#' + id) history.replaceState(null, '', '#' + id);
    if (scroll) window.scrollTo({top:0, behavior:'instant'});
  }
  document.querySelectorAll('[data-page]').forEach(b => b.addEventListener('click', () => navigate(b.dataset.page)));
  document.querySelectorAll('[data-go]').forEach(b => b.addEventListener('click', () => navigate(b.dataset.go)));
  window.addEventListener('hashchange', () => navigate(location.hash.slice(1)));
  $('competition-facts').textContent = 'Kaggle · Featured Code Competition · Final deadline 14 Jan 2027, 23:59 UTC';
  $('snapshot-time').textContent = 'Snapshot: ' + stamp;
  $('footer-time').textContent = stamp + ' · not live';
  $('rank-note').textContent = 'Kaggle reported rank ' + D.competition.user_rank.toLocaleString() + ' among ' + D.competition.team_count.toLocaleString() + ' teams at this snapshot (' + stamp + ').';
  $('training-snapshot').textContent = D.training.name + ' status: ' + D.training.status + ', checked ' + new Date(D.training.checked_at).toLocaleString('en-GB',{timeZone:'UTC'}) + ' UTC. Runtime estimate was about five hours; no completion or score is implied.';
  $('data-hash').textContent = 'Training file SHA-256: ' + D.provenance.host_sha256;

  options($('model-filter'), models); options($('prompt-filter'), prompts);
  let filtered = D.rows; let position = 0;
  function renderComparison() {
    if (!filtered.length) return;
    const row = filtered[position];
    const other = promptRows.get(row.prompt_id).find(r => r.model === $('compare-model').value);
    if (!other) { $('compare-text').textContent = 'No row for this model and prompt.'; $('compare-meta').textContent = ''; return; }
    $('compare-text').textContent = other.text;
    $('compare-meta').textContent = other.id + ' · ' + other.prompt_id + ' · ' + other.words + ' words';
  }
  function renderSample() {
    $('sample-position').textContent = filtered.length ? 'Passage ' + (position + 1).toLocaleString() + ' of ' + filtered.length.toLocaleString() + ' matches' : '0 matching passages';
    $('previous').disabled = position === 0 || !filtered.length;
    $('next').disabled = position >= filtered.length - 1;
    $('reader-content').hidden = !filtered.length; $('empty-data').hidden = !!filtered.length;
    if (!filtered.length) return;
    const row = filtered[position];
    $('sample-model').textContent = row.model;
    $('sample-meta').textContent = row.id + ' · ' + row.prompt_id + ' · ' + row.words + ' words';
    $('sample-text').textContent = row.text;
    const prior = $('compare-model').value;
    const choices = promptRows.get(row.prompt_id).filter(r => r.model !== row.model).map(r => r.model).sort();
    $('compare-model').replaceChildren(); options($('compare-model'), choices);
    if (choices.includes(prior)) $('compare-model').value = prior;
    renderComparison();
  }
  function filterRows() {
    const q = $('search').value.trim().toLocaleLowerCase(), model = $('model-filter').value, prompt = $('prompt-filter').value;
    filtered = D.rows.filter(r => (!model || r.model === model) && (!prompt || r.prompt_id === prompt) && (!q || (r.text + ' ' + r.id + ' ' + r.prompt_id).toLocaleLowerCase().includes(q)));
    position = 0; renderSample();
  }
  $('search').addEventListener('input', filterRows);
  ['model-filter','prompt-filter'].forEach(id => $(id).addEventListener('change', filterRows));
  $('compare-model').addEventListener('change', renderComparison);
  $('previous').addEventListener('click', () => { if (position > 0) { position--; renderSample(); } });
  $('next').addEventListener('click', () => { if (position < filtered.length - 1) { position++; renderSample(); } });
  $('reset-data').addEventListener('click', () => { ['search','model-filter','prompt-filter'].forEach(id => $(id).value = ''); filterRows(); });
  models.forEach(name => { const row = el('div', undefined, 'model-row'); row.append(el('span',name),el('b',String(D.model_counts[name]))); $('model-counts').append(row); });
  D.visible_test.forEach(row => { const d = el('details'); d.append(el('summary','Test ID ' + row.id + ' · label hidden'), el('p',row.text,'passage-text')); $('test-rows').append(d); });
  renderSample();

  const versions = {57438413:1,57439219:2,57440737:3,57444632:4,57491848:7,57517925:8,57523017:10,57600315:16,57632200:21,57655346:23};
  const submissions = [...D.submissions].filter(s => s.status === 'COMPLETE' && s.public_score !== undefined).sort((a,b) => new Date(a.date)-new Date(b.date));
  submissions.forEach((s,i) => {
    const version = 'V' + versions[s.ref], score = Number(s.public_score);
    const row = el('div',undefined,'score-bar-row');
    const track = el('div',undefined,'bar-track'); const fill = el('div',undefined,'bar-fill' + (version==='V16'?' credited':version==='V21'?' best':''));
    fill.style.width = (score / .7 * 100) + '%'; track.append(fill);
    row.append(el('span',version),track,el('b',score.toFixed(3))); $('score-chart').append(row);
    const tr = el('tr'); const delta = i ? score - Number(submissions[i-1].public_score) : null;
    tr.append(el('td',version),el('td',new Date(s.date).toISOString().slice(0,16).replace('T',' ')),el('td',score.toFixed(3)),el('td',delta===null?'—':(delta>=0?'+':'')+delta.toFixed(3),delta===null?'':delta>=0?'positive':'negative'));
    const td=el('td'); const a=el('a','Open ↗'); a.href='https://www.kaggle.com'+s.url; a.target='_blank'; a.rel='noopener'; td.append(a);tr.append(td);$('score-table').append(tr);
  });
  $('score-chart').append(el('div','Scale: 0.000 → 0.700 target · grey: credited V16 · gold: best V21','score-target'));

  options($('family-filter'), [...new Set(D.experiments.map(e => e.family))].sort());
  function renderExperiments() {
    const status=$('status-filter').value, family=$('family-filter').value, query=$('experiment-search').value.trim().toLowerCase();
    const matches=D.experiments.filter(e => (!status||e.status===status)&&(!family||e.family===family)&&(!query||('V'+e.version+' '+e.title+' '+e.method+' '+e.why+' '+e.outcome).toLowerCase().includes(query)));
    $('experiment-count').textContent=matches.length+' of '+D.experiments.length+' experiment records';
    $('experiment-list').replaceChildren();
    matches.forEach(e => {
      const detail=el('details',undefined,'experiment'); detail.id='experiment-'+e.version;
      const summary=el('summary');const title=el('span',e.title,'experiment-name');title.append(el('small',e.family));
      const statusClass=e.status==='Did not qualify'?'failed':e.status==='Planned'?'planned':e.status==='Running'?'running':'';
      summary.append(el('span','V'+e.version,'version'),title,el('span',e.status,'status '+statusClass));
      const body=el('div',undefined,'experiment-body');
      [['What we did',e.method],['Why we tried it',e.why],['Local evidence — not a public score',e.local],['Outcome',e.outcome]].forEach(([label,text])=>{const part=el('div',undefined,label.startsWith('Local')?'local-result':'');part.append(el('h3',label),el('p',text));body.append(part);});
      const link=el('a','Read the full recorded history ↗'); link.href='technical-history.txt';link.target='_blank';link.rel='noopener';body.append(link);
      detail.append(summary,body);$('experiment-list').append(detail);
    });
  }
  ['status-filter','family-filter'].forEach(id => $(id).addEventListener('change',renderExperiments));
  $('experiment-search').addEventListener('input',renderExperiments);
  renderExperiments();navigate(location.hash.slice(1),false);
})();
