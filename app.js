// app.js
// Single shared script for 3-page GitHub Pages AHP tool (setup, matrices, results)
// Pages must have <body data-page="setup|matrices|results"> and a <div id="view"></div>

const STORAGE_KEY = "ahp_state_pages_v1";

// Random Index (Saaty)
const RI = { 1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49 };

// -------------------- State --------------------
function defaultState(){
  const st = {
    problem: { name: "Logistics", goal: "Select the best warehouse location" },
    criteria: ["Transportation cost", "Delivery lead time", "Service reliability"],
    alternatives: ["Location A", "Location B", "Location C"],

    criteriaMatrix: [],
    altMatrices: [],

    activeCritIdx: 0
  };
  initMatrices(st);
  return st;
}

function identityMatrix(n){
  return Array.from({length:n}, (_,i)=>Array.from({length:n}, (_,j)=> i===j ? 1 : 1));
}

function initMatrices(st){
  st.criteriaMatrix = identityMatrix(st.criteria.length);
  st.altMatrices = st.criteria.map(()=> identityMatrix(st.alternatives.length));
}

function loadState(){
  const raw = localStorage.getItem(STORAGE_KEY);
  if(!raw) return defaultState();

  try{
    const st = JSON.parse(raw);
    if(!st.problem || !Array.isArray(st.criteria) || !Array.isArray(st.alternatives)) return defaultState();

    if(!Array.isArray(st.criteriaMatrix) || !Array.isArray(st.altMatrices)) initMatrices(st);

    if(typeof st.activeCritIdx !== "number") st.activeCritIdx = 0;
    if(st.activeCritIdx < 0) st.activeCritIdx = 0;
    if(st.activeCritIdx >= st.criteria.length) st.activeCritIdx = 0;

    // Ensure matrices match current sizes, else reset them
    if(st.criteriaMatrix.length !== st.criteria.length) initMatrices(st);
    if(st.altMatrices.length !== st.criteria.length) initMatrices(st);
    if(st.altMatrices.some(m => !Array.isArray(m) || m.length !== st.alternatives.length)) initMatrices(st);

    return st;
  }catch{
    return defaultState();
  }
}

function saveState(st){
  localStorage.setItem(STORAGE_KEY, JSON.stringify(st));
}

// -------------------- Math helpers --------------------
function cloneMatrix(A){ return A.map(r=>r.slice()); }

function setPairwise(A, i, j, v){
  const B = cloneMatrix(A);
  B[i][j] = v;
  B[j][i] = 1 / v;
  for(let k=0;k<B.length;k++) B[k][k] = 1;
  return B;
}

// power iteration to approximate principal eigenvector
function powerIterationWeights(A, maxIter=1500, tol=1e-11){
  const n = A.length;
  let w = Array(n).fill(1/n);

  for(let it=0; it<maxIter; it++){
    const Aw = Array(n).fill(0);
    for(let i=0;i<n;i++){
      let s=0;
      for(let j=0;j<n;j++) s += A[i][j] * w[j];
      Aw[i]=s;
    }
    const sum = Aw.reduce((a,b)=>a+b,0);
    const wNew = Aw.map(x=>x/sum);

    let diff = 0;
    for(let i=0;i<n;i++) diff = Math.max(diff, Math.abs(wNew[i]-w[i]));
    w = wNew;

    if(diff < tol) break;
  }

  const Aw2 = Array(n).fill(0);
  for(let i=0;i<n;i++){
    let s=0;
    for(let j=0;j<n;j++) s += A[i][j] * w[j];
    Aw2[i]=s;
  }
  const lambdaMax = Aw2.reduce((a,v,i)=> a + v / w[i], 0) / n;

  return { weights: w, lambdaMax };
}

function consistency(A, lambdaMax){
  const n = A.length;
  const ci = (lambdaMax - n) / (n - 1);
  const ri = RI[n] ?? 1.49;
  const cr = (ri === 0) ? 0 : (ci / ri);
  return { ci, cr };
}

function ahpSolve(A){
  const { weights, lambdaMax } = powerIterationWeights(A);
  const { ci, cr } = consistency(A, lambdaMax);
  return { weights, lambdaMax, ci, cr };
}

// -------------------- UI helpers --------------------
function escapeHtml(s){
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;");
}

function setStatus(st){
  const el = document.getElementById("status");
  if(!el) return;
  el.textContent = `Criteria: ${st.criteria.length}, Alternatives: ${st.alternatives.length}`;
}

function crMessage(cr){
  if(cr <= 0.10){
    return { level: "good", title: "Consistenza buona", text: `CR ${cr.toFixed(3)}. I confronti sono coerenti.` };
  }
  if(cr <= 0.20){
    return { level: "mid", title: "Consistenza borderline", text: `CR ${cr.toFixed(3)}. Rivedi 1 o 2 confronti.` };
  }
  return { level: "warn", title: "Consistenza bassa", text: `CR ${cr.toFixed(3)}. Il ranking può cambiare. Rivedi i confronti suggeriti.` };
}

function crBadge(cr){
  const m = crMessage(cr);
  const cls = m.level === "good" ? "badge good" : (m.level === "mid" ? "badge mid" : "badge warn");
  return `<span class="${cls}">${escapeHtml(m.title)}. ${escapeHtml(m.text)}</span>`;
}

function inconsistencyHints(labels, A, topK=3){
  const n = labels.length;
  const issues = [];

  for(let i=0;i<n;i++){
    for(let j=i+1;j<n;j++){
      const direct = A[i][j];
      if(!Number.isFinite(direct) || direct <= 0) continue;

      let best = 0;
      let bestK = -1;

      for(let k=0;k<n;k++){
        if(k === i || k === j) continue;
        const via = A[i][k] * A[k][j];
        if(!Number.isFinite(via) || via <= 0) continue;

        const d = Math.abs(Math.log(direct) - Math.log(via));
        if(d > best){
          best = d;
          bestK = k;
        }
      }

      if(bestK >= 0){
        issues.push({
          score: best,
          text: `${labels[i]} vs ${labels[j]} (check with ${labels[bestK]})`
        });
      }
    }
  }

  issues.sort((a,b)=>b.score-a.score);
  return issues.slice(0, topK);
}

function hintsHtml(hints){
  if(!hints || hints.length === 0) return "";
  return `
    <div style="margin-top:10px;">
      <div class="small muted" style="margin-bottom:6px;">Suggested checks</div>
      <ul style="margin:0; padding-left: 18px; color: var(--muted); font-size: 13px;">
        ${hints.map(h=>`<li>${escapeHtml(h.text)}</li>`).join("")}
      </ul>
    </div>
  `;
}

// -------------------- Heatmap --------------------
function clamp(x,a,b){ return Math.max(a, Math.min(b, x)); }
function lerp(a,b,t){ return a + (b - a) * t; }

function mixColor(c1, c2, t){
  return [
    Math.round(lerp(c1[0], c2[0], t)),
    Math.round(lerp(c1[1], c2[1], t)),
    Math.round(lerp(c1[2], c2[2], t)),
  ];
}

function rgb(arr){ return `rgb(${arr[0]},${arr[1]},${arr[2]})`; }

function matrixHeatmap(containerId, title){
  return `
    <div class="heatWrap">
      <div class="heatTitle">
        <div class="panelTitle" style="margin:0;">${escapeHtml(title)}</div>
        <div class="heatLegend">
          <span>low</span>
          <span class="heatLegendBar"></span>
          <span>high</span>
        </div>
      </div>
      <canvas class="heatCanvas" id="${containerId}" width="900" height="560"></canvas>
    </div>
  `;
}

function drawMatrixHeatmap(canvasId, labels, A){
  const c = document.getElementById(canvasId);
  if(!c) return;
  const ctx = c.getContext("2d");

  const n = labels.length;
  const W = c.width;
  const H = c.height;
  ctx.clearRect(0,0,W,H);

  const pad = 16;
  const top = 62;
  const left = 170;

  const sizeW = W - left - pad;
  const sizeH = H - top - pad;
  const cell = Math.floor(Math.min(sizeW, sizeH) / n);

  let maxAbs = 0.0;
  for(let i=0;i<n;i++){
    for(let j=0;j<n;j++){
      const v = Number(A[i][j]);
      if(!Number.isFinite(v) || v <= 0) continue;
      const lv = Math.log(v);
      maxAbs = Math.max(maxAbs, Math.abs(lv));
    }
  }
  if(maxAbs === 0) maxAbs = 1;

  const blue = [29, 78, 216];
  const white = [255, 255, 255];
  const red = [220, 38, 38];

  ctx.fillStyle = "rgba(15,23,42,0.88)";
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for(let j=0;j<n;j++){
    const x = left + j*cell + cell/2;
    ctx.save();
    ctx.translate(x, top - 18);
    ctx.rotate(-0.35);
    ctx.fillText(labels[j], 0, 0);
    ctx.restore();
  }

  ctx.textAlign = "right";
  for(let i=0;i<n;i++){
    const y = top + i*cell + cell/2;
    ctx.fillText(labels[i], left - 10, y);
  }

  ctx.textAlign = "center";
  for(let i=0;i<n;i++){
    for(let j=0;j<n;j++){
      const v = Number(A[i][j]);
      const x = left + j*cell;
      const y = top + i*cell;

      let col = white;
      if(Number.isFinite(v) && v > 0){
        const t = clamp(Math.log(v) / maxAbs, -1, 1);
        if(t < 0) col = mixColor(white, blue, Math.abs(t));
        if(t > 0) col = mixColor(white, red, Math.abs(t));
      }

      ctx.fillStyle = rgb(col);
      ctx.fillRect(x, y, cell, cell);

      ctx.strokeStyle = "rgba(15,23,42,0.10)";
      ctx.strokeRect(x, y, cell, cell);

      ctx.fillStyle = "rgba(15,23,42,0.86)";
      ctx.font = "11px system-ui";
      const txt = Number.isFinite(v) ? v.toFixed(2) : "";
      ctx.fillText(txt, x + cell/2, y + cell/2);
    }
  }
}

// -------------------- Charts --------------------
function drawBarChart(canvasId, title, items){
  const c = document.getElementById(canvasId);
  if(!c) return;
  const ctx = c.getContext("2d");

  const W = c.width;
  const H = c.height;
  ctx.clearRect(0,0,W,H);

  ctx.fillStyle = "rgba(15,23,42,0.92)";
  ctx.font = "18px system-ui";
  ctx.fillText(title, 18, 28);

  const maxV = Math.max(...items.map(x=>x.value), 0.00001);
  const left = 18;
  const right = 18;
  const top = 44;
  const bottom = 18;

  const chartW = W - left - right;
  const chartH = H - top - bottom;

  ctx.strokeStyle = "rgba(15,23,42,0.16)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top + chartH);
  ctx.lineTo(left + chartW, top + chartH);
  ctx.stroke();

  const n = items.length;
  const gap = 16;
  const barW = (chartW - gap*(n-1)) / n;

  items.forEach((x, i)=>{
    const h = (x.value / maxV) * (chartH * 0.92);
    const bx = left + i*(barW+gap);
    const by = top + chartH - h;

    ctx.fillStyle = "rgba(37,99,235,0.18)";
    ctx.fillRect(bx, by, barW, h);

    ctx.fillStyle = "rgba(15,23,42,0.76)";
    ctx.font = "12px system-ui";
    ctx.fillText(x.name, bx, top + chartH + 14);

    ctx.fillStyle = "rgba(15,23,42,0.90)";
    ctx.font = "12px system-ui";
    ctx.fillText(x.value.toFixed(3), bx, by - 6);
  });
}

// -------------------- Controls: pairwise UI --------------------
function pairwiseUI(labels, A){
  const n = labels.length;
  let html = "";

  for(let i=0;i<n;i++){
    for(let j=i+1;j<n;j++){
      const aij = A[i][j];

      let v = aij;
      if(v < 1) v = 1 / v;
      v = Math.min(9, Math.max(1, Math.round(v)));

      html += `
        <div class="pairRow" data-i="${i}" data-j="${j}">
          <div>${escapeHtml(labels[i])}</div>
          <div class="pairMid">
            <div class="sidePick">
              <button type="button" class="pickLeft">Left</button>
              <button type="button" class="pickRight">Right</button>
            </div>
            <input type="range" class="rng" min="1" max="9" step="1" value="${v}" />
            <div class="valBox" style="width:22px; text-align:right">${v}</div>
          </div>
          <div style="text-align:right">${escapeHtml(labels[j])}</div>
        </div>
      `;
    }
  }

  return html;
}

function bindPairwiseHandlers(rootEl, labels, A, onUpdate){
  if(!rootEl) return;

  rootEl.querySelectorAll(".pairRow").forEach(row=>{
    const i = Number(row.dataset.i);
    const j = Number(row.dataset.j);

    const rng = row.querySelector(".rng");
    const valBox = row.querySelector(".valBox");
    const pickLeft = row.querySelector(".pickLeft");
    const pickRight = row.querySelector(".pickRight");

    if(!rng || !valBox || !pickLeft || !pickRight) return;

    let preferRight = A[i][j] < 1;

    const syncButtons = ()=>{
      pickLeft.classList.toggle("active", !preferRight);
      pickRight.classList.toggle("active", preferRight);
    };

    const commit = ()=>{
      const raw = Number(rng.value);
      valBox.textContent = String(raw);

      const val = preferRight ? 1/raw : raw;
      const B = setPairwise(A, i, j, val);
      onUpdate(B);
    };

    rng.addEventListener("input", ()=>{
      valBox.textContent = String(rng.value);
    });

    rng.addEventListener("change", commit);

    pickLeft.addEventListener("click", ()=>{
      preferRight = false;
      syncButtons();
      commit();
    });

    pickRight.addEventListener("click", ()=>{
      preferRight = true;
      syncButtons();
      commit();
    });

    syncButtons();
  });
}


// -------------------- Page rendering --------------------
function renderEditableList(containerId, items, onChange, minLen){
  const root = document.getElementById(containerId);
  if(!root) return;

  root.innerHTML = "";
  items.forEach((v, i)=>{
    const row = document.createElement("div");
    row.className = "kv";
    row.style.marginBottom = "8px";

    const input = document.createElement("input");
    input.type = "text";
    input.value = v;
    input.addEventListener("input", (e)=>{
      const next = items.slice();
      next[i] = e.target.value;
      onChange(next);
    });

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn";
    btn.textContent = "-";
    btn.disabled = items.length <= minLen;

    btn.addEventListener("click", ()=>{
      if(items.length <= minLen) return;
      const next = items.filter((_,k)=>k!==i);
      onChange(next);
    });

    row.appendChild(input);
    row.appendChild(btn);
    root.appendChild(row);
  });
}

function renderSetupPage(st){
  const view = document.getElementById("view");
  if(!view) return;

  view.innerHTML = `
    <div class="row">
      <div>
        <div class="panelTitle">Problem</div>
        <label class="small muted">Name</label>
        <input id="p_name" type="text" value="${escapeHtml(st.problem.name)}" />
        <div style="height:10px"></div>
        <label class="small muted">Goal</label>
        <input id="p_goal" type="text" value="${escapeHtml(st.problem.goal)}" />
      </div>

      <div>
        <div class="panelTitle">Scale</div>
        <div class="badge">1 equal, 3 moderate, 5 strong, 7 major, 9 extreme</div>
        <div style="height:12px"></div>
        <div class="small muted">Fill criteria and alternatives, then go to matrices.</div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="row">
      <div>
        <div class="panelTitle">Criteria</div>
        <div id="crit_list"></div>
        <button type="button" class="btn inline" id="crit_add">Add criterion</button>
      </div>

      <div>
        <div class="panelTitle">Alternatives</div>
        <div id="alt_list"></div>
        <button type="button" class="btn inline" id="alt_add">Add alternative</button>
      </div>
    </div>
  `;

  const pName = document.getElementById("p_name");
  const pGoal = document.getElementById("p_goal");

  pName.addEventListener("input", e=>{
    st.problem.name = e.target.value;
    saveState(st);
  });

  pGoal.addEventListener("input", e=>{
    st.problem.goal = e.target.value;
    saveState(st);
  });

  renderEditableList("crit_list", st.criteria, (arr)=>{
    st.criteria = arr;
    st.activeCritIdx = 0;
    initMatrices(st);
    saveState(st);
    renderSetupPage(st);
    setStatus(st);
  }, 2);

  renderEditableList("alt_list", st.alternatives, (arr)=>{
    st.alternatives = arr;
    initMatrices(st);
    saveState(st);
    renderSetupPage(st);
    setStatus(st);
  }, 2);

  document.getElementById("crit_add").addEventListener("click", ()=>{
    st.criteria.push(`C${st.criteria.length + 1}`);
    initMatrices(st);
    saveState(st);
    renderSetupPage(st);
    setStatus(st);
  });

  document.getElementById("alt_add").addEventListener("click", ()=>{
    st.alternatives.push(`A${st.alternatives.length + 1}`);
    initMatrices(st);
    saveState(st);
    renderSetupPage(st);
    setStatus(st);
  });
}

function renderMatricesPage(st){
  const view = document.getElementById("view");
  if(!view) return;

  // criteria
  const critSolve = ahpSolve(st.criteriaMatrix);
  const critHints = inconsistencyHints(st.criteria, st.criteriaMatrix, 3);

  // active criterion for alt comparisons
  let activeIdx = st.activeCritIdx ?? 0;
  if(activeIdx < 0 || activeIdx >= st.criteria.length) activeIdx = 0;
  st.activeCritIdx = activeIdx;

  const A = st.altMatrices[activeIdx];
  const altSolve = ahpSolve(A);
  const altHints = inconsistencyHints(st.alternatives, A, 3);

  view.innerHTML = `
    <div class="panelTitle">Criteria comparisons</div>
    <div class="small muted">${crBadge(critSolve.cr)}${critSolve.cr > 0.10 ? hintsHtml(critHints) : ""}</div>
    <div style="height:10px"></div>

    <div class="matrixLayout">
      <div class="stickyBox">
        ${matrixHeatmap("hm_crit", "Criteria matrix")}
        <div style="display:flex; gap:10px; margin-top:10px;">
          <button type="button" class="btn inline" id="crit_reset">Reset criteria</button>
        </div>
      </div>
      <div id="critPairs"></div>
    </div>

    <div class="divider"></div>

    <div class="panelTitle">Alternatives by criterion</div>
    <div class="tabs" id="critTabs"></div>
    <div class="small muted">${crBadge(altSolve.cr)}${altSolve.cr > 0.10 ? hintsHtml(altHints) : ""}</div>
    <div style="height:10px"></div>

    <div class="matrixLayout">
      <div class="stickyBox">
        ${matrixHeatmap("hm_alt", "Alternatives matrix")}
        <div style="display:flex; gap:10px; margin-top:10px;">
          <button type="button" class="btn inline" id="alt_reset">Reset this matrix</button>
        </div>
      </div>
      <div id="altPairs"></div>
    </div>
  `;

  // pairwise blocks
const critPairsEl = document.getElementById("critPairs");
critPairsEl.innerHTML = pairwiseUI(st.criteria, st.criteriaMatrix);

bindPairwiseHandlers(critPairsEl, st.criteria, st.criteriaMatrix, (B)=>{
  st.criteriaMatrix = B;
  saveState(st);
  renderMatricesPage(st);
  setStatus(st);
});


  document.getElementById("altPairs").innerHTML = pairwiseUI(st.alternatives, A, (B)=>{
    st.altMatrices[activeIdx] = B;
    saveState(st);
    renderMatricesPage(st);
    setStatus(st);
  });

  // tabs for selecting criterion
  const tabs = document.getElementById("critTabs");
  tabs.innerHTML = "";
  st.criteria.forEach((c, i)=>{
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tabBtn" + (i === activeIdx ? " active" : "");
    b.textContent = c;
    b.addEventListener("click", ()=>{
      st.activeCritIdx = i;
      saveState(st);
      renderMatricesPage(st);
      setStatus(st);
    });
    tabs.appendChild(b);
  });

  // reset buttons
  document.getElementById("crit_reset").addEventListener("click", ()=>{
    st.criteriaMatrix = identityMatrix(st.criteria.length);
    saveState(st);
    renderMatricesPage(st);
    setStatus(st);
  });

  document.getElementById("alt_reset").addEventListener("click", ()=>{
    st.altMatrices[activeIdx] = identityMatrix(st.alternatives.length);
    saveState(st);
    renderMatricesPage(st);
    setStatus(st);
  });

  // draw heatmaps after DOM
  setTimeout(()=>{
    drawMatrixHeatmap("hm_crit", st.criteria, st.criteriaMatrix);
    drawMatrixHeatmap("hm_alt", st.alternatives, st.altMatrices[activeIdx]);
  }, 0);
}

function computeResults(st){
  const crit = ahpSolve(st.criteriaMatrix);
  const altSolves = st.altMatrices.map(A => ahpSolve(A));

  const m = st.alternatives.length;
  const n = st.criteria.length;

  const scores = Array(m).fill(0);
  for(let i=0;i<m;i++){
    let s=0;
    for(let j=0;j<n;j++){
      s += crit.weights[j] * altSolves[j].weights[i];
    }
    scores[i]=s;
  }

  const ranking = st.alternatives
    .map((name,i)=>({ name, score: scores[i] }))
    .sort((a,b)=>b.score-a.score);

  return {
    problem: st.problem,
    criteria: st.criteria,
    alternatives: st.alternatives,
    results: {
      criteriaWeights: crit.weights,
      criteriaCR: crit.cr,
      altWeightsByCriterion: st.criteria.reduce((acc, c, i)=>{
        acc[c] = altSolves[i].weights;
        return acc;
      }, {}),
      altCRByCriterion: st.criteria.reduce((acc, c, i)=>{
        acc[c] = altSolves[i].cr;
        return acc;
      }, {}),
      finalScores: scores,
      ranking
    }
  };
}

function rankingTable(items){
  let html = `<table><thead><tr><th>Alternative</th><th>Score</th></tr></thead><tbody>`;
  items.forEach(x=>{
    html += `<tr><td>${escapeHtml(x.name)}</td><td>${x.score.toFixed(6)}</td></tr>`;
  });
  html += `</tbody></table>`;
  return html;
}

function weightsTable(labels, weights){
  return `
    <table>
      <thead><tr><th>Item</th><th>Weight</th></tr></thead>
      <tbody>
        ${labels.map((x,i)=>`<tr><td>${escapeHtml(x)}</td><td>${Number(weights[i]).toFixed(6)}</td></tr>`).join("")}
      </tbody>
    </table>
  `;
}

function altWeightsTables(res){
  let html = `<div class="row">`;
  res.criteria.forEach((c)=>{
    const w = res.results.altWeightsByCriterion[c];
    const cr = res.results.altCRByCriterion[c];
    html += `
      <div>
        <div class="small muted">${escapeHtml(c)} ${crBadge(cr)}</div>
        <div style="height:8px"></div>
        ${weightsTable(res.alternatives, w)}
      </div>
    `;
  });
  html += `</div>`;
  return html;
}

function renderResultsPage(st){
  const view = document.getElementById("view");
  if(!view) return;

  const res = computeResults(st);
  const best = res.results.ranking[0];

  view.innerHTML = `
    <div class="row">
      <div>
        <div class="panelTitle">Key results</div>
        <div style="font-size:18px; font-weight:700; margin-top:6px">${escapeHtml(best.name)}</div>
        <div class="small muted">Top ranked alternative. Score ${best.score.toFixed(4)}</div>
        <div style="height:10px"></div>
        ${crBadge(res.results.criteriaCR)}
        <div class="divider"></div>
        <div class="panelTitle">Ranking</div>
        ${rankingTable(res.results.ranking)}
      </div>

      <div>
        <div class="panelTitle">Charts</div>
        <canvas class="chart" id="chartCrit" width="900" height="320"></canvas>
        <div style="height:12px"></div>
        <canvas class="chart" id="chartScore" width="900" height="320"></canvas>
      </div>
    </div>

    <div class="divider"></div>

    <div class="panelTitle">Criteria weights</div>
    ${weightsTable(res.criteria, res.results.criteriaWeights)}

    <div class="divider"></div>

    <div class="panelTitle">Alternative weights by criterion</div>
    ${altWeightsTables(res)}
  `;

  const critRows = res.criteria.map((c, i)=>({ name: c, value: res.results.criteriaWeights[i] }));
  const scoreRows = res.alternatives.map((a, i)=>({ name: a, value: res.results.finalScores[i] }));

  setTimeout(()=>{
    drawBarChart("chartCrit", "Criteria weights", critRows);
    drawBarChart("chartScore", "Final scores", scoreRows);
  }, 0);
}

// -------------------- Navigation and actions --------------------
function wireNavButtons(st){
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");

  if(prevBtn){
    prevBtn.addEventListener("click", ()=>{
      saveState(st);
      const page = document.body.dataset.page;
      if(page === "matrices") location.href = "setup.html";
      if(page === "results") location.href = "matrices.html";
    });
  }

  if(nextBtn){
    nextBtn.addEventListener("click", ()=>{
      saveState(st);
      const page = document.body.dataset.page;
      if(page === "setup") location.href = "matrices.html";
      if(page === "matrices") location.href = "results.html";
    });
  }
}

function wireCommonButtons(st){
  const reset = document.getElementById("btnReset");
  if(reset){
    reset.addEventListener("click", ()=>{
      const fresh = defaultState();
      saveState(fresh);
      location.href = "setup.html";
    });
  }

  const exp = document.getElementById("btnExport");
  if(exp){
    exp.addEventListener("click", ()=>{
      const payload = computeResults(st);
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "ahp_results.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  }
}

// -------------------- Main --------------------
function main(){
  const st = loadState();

  wireNavButtons(st);
  wireCommonButtons(st);

  setStatus(st);

  const page = document.body.dataset.page;
  const view = document.getElementById("view");
  if(!view) return;

  if(page === "setup") renderSetupPage(st);
  if(page === "matrices") renderMatricesPage(st);
  if(page === "results") renderResultsPage(st);

  saveState(st);
}

document.addEventListener("DOMContentLoaded", main);
