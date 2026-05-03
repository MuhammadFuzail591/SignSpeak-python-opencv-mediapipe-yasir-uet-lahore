let camRunning = false;
let pollTimer  = null;
let history    = [];
let lastAudio  = null;

// ---- Camera ----
async function startCamera() {
  setBtnState('btn-start', false, '<span class="spin">⟳</span> Starting…');
  try {
    const r = await fetch('/start', { method: 'POST' });
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    camRunning = true;
    const feed = document.getElementById('video-feed');
    feed.src = '/video_feed';
    feed.style.display = 'block';
    document.getElementById('placeholder').classList.add('hidden');

    setPill('live', 'LIVE');
    setBtnState('btn-start', true, '▶ Start Camera');
    setBtnState('btn-stop',  false, '■ Stop');
    setBtnState('btn-trans', false, '🤟 Translate');

    pollTimer = setInterval(pollHand, 800);
  } catch(e) {
    showToast(e.message || 'Failed to start camera');
    setBtnState('btn-start', false, '▶ Start Camera');
  }
}

async function stopCamera() {
  await fetch('/stop', { method: 'POST' });
  camRunning = false;
  clearInterval(pollTimer);

  const feed = document.getElementById('video-feed');
  feed.style.display = 'none';
  feed.src = '';
  document.getElementById('placeholder').classList.remove('hidden');

  setPill('', 'OFFLINE');
  setBtnState('btn-start', false, '▶ Start Camera');
  setBtnState('btn-stop',  true,  '■ Stop');
  setBtnState('btn-trans', true,  '🤟 Translate');
}

async function pollHand() {
  try {
    const r = await fetch('/hand_status');
    const d = await r.json();
    setPill(d.hand_detected ? 'hand' : 'live',
            d.hand_detected ? 'HAND DETECTED' : 'LIVE');
  } catch {}
}

// ---- Translate ----
async function translate_func() {
  if (!camRunning) return;
  console.log("Translate function running")
  const speed = document.querySelector('input[name="spd"]:checked').value;
  const box   = document.getElementById('result-box');

  box.className = 'result-box loading';
  document.getElementById('rval').style.display = 'none';
  document.getElementById('rph').style.display  = 'none';
  setBtnState('btn-trans', true, '<span class="spin">⟳</span> Translating…');

  try {
    const r = await fetch('/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    box.className = 'result-box active';
    document.getElementById('rph').style.display  = 'none';
    const rv = document.getElementById('rval');
    rv.textContent = d.text;
    rv.style.display = 'inline';

    if (d.audio) {
      lastAudio = d.audio;
      const p = document.getElementById('player');
      p.src = d.audio;
      p.play().catch(()=>{});
      document.getElementById('audio-row').style.display = 'flex';
    }

    addHistory(d.text);
  } catch(e) {
    box.className = 'result-box';
    document.getElementById('rph').style.display = 'inline';
    document.getElementById('rph').textContent   = 'Translation failed';
    showToast(e.message || 'Error during translation');
  } finally {
    setBtnState('btn-trans', false, '🤟 Translate');
  }
}

function replayAudio() {
  if (!lastAudio) return;
  const p = document.getElementById('player');
  p.src = lastAudio;
  p.play().catch(()=>{});
}

function copyResult() {
  const v = document.getElementById('rval').textContent;
  if (v) navigator.clipboard.writeText(v).then(()=> showToast('Copied!'));
}

// ---- History ----
function addHistory(text) {
  const t = new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  history.unshift({ text, t });
  if (history.length > 30) history.pop();
  renderHistory();
}

function renderHistory() {
  const el = document.getElementById('history');
  if (!history.length) {
    el.innerHTML = '<div class="h-empty">No translations yet</div>';
    return;
  }
  el.innerHTML = history.map(h =>
    `<div class="h-item" onclick="speakText('${h.text.replace(/'/g,"\\'")}')">
      <span>${h.text}</span>
      <span class="h-time">${h.t}</span>
     </div>`
  ).join('');
}

function clearHistory() {
  history = [];
  renderHistory();
}

function speakText(txt) {
  if ('speechSynthesis' in window) {
    const u = new SpeechSynthesisUtterance(txt);
    u.rate = 0.9;
    speechSynthesis.speak(u);
  }
}

// ---- Helpers ----
function setPill(type, text) {
  const el = document.getElementById('cam-pill');
  el.className = 'cam-pill ' + type;
  document.getElementById('pill-text').textContent = text;
}

function setBtnState(id, disabled, html) {
  const el = document.getElementById(id);
  el.disabled = disabled;
  el.innerHTML = html;
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 4000);
}

// Keyboard shortcut
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && camRunning && !e.target.matches('input,button,textarea')) {
    e.preventDefault();
    translate_func();
  }
});

// Init history
renderHistory();
