// ---------- State ----------
const state = {
  step: 1,
  problem: { name: "Logistics", goal: "Select the best warehouse location" },
  criteria: ["Transportation cost", "Delivery lead time", "Service reliability"],
  alternatives: ["Location A", "Location B", "Location C"],

  criteriaMatrix: [],
  altMatrices: [], // one matrix per criterion
};

const RI = { 1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49 };

function identityMatrix(n){
  return Array.from({length:n}, (_,i)=>Array.from({length:n}, (_,j)=> i===j ? 1 : 1));
}

function cloneMatrix(A){ return A.map(r=>r.slice()); }

function setPairwise(A, i, j, v){
  const B = cloneMatrix(A);
  B[i][j] = v;
  B[j][i] = 1 / v;
  for(let k=0;k<B.length;k++) B[k][k] = 1;
  return B;
}

function powerIterationWeights(A, maxIter=1000, tol=1e-10){
  const n = A.length;
  let w = Array(n).fill(1/n);

  for(let it=0; it<maxIter; it++){
    const Aw = Array(n).fill(0);
    for(let i=0;i<n;i++){
      let s=0;
      for(let j=0;j<n;j++) s += A[i][j]*w[j];
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
    for(let j=0;j<n;j++) s += A[i][j]*w[j];
    Aw2[i]=s;
  }
  const lambdaMax = Aw2.reduce((a,v,i)=>a + v / w[i], 0) / n;

  return { weights: w, lambdaMax };
}

function consistency(A, lambdaMax){
  const n = A.length;
  const ci = (lambdaMax - n) / (n - 1);
  const ri = RI[n] ?? 1.49;
  const cr = ri === 0 ? 0 : ci / ri;
  return { ci, cr };
}

function ahpSolve(A){
  const { weights, lambdaMax } = powerIterationWeights(A);
  const { ci, cr } = consistency(A, lambdaMax);
  return { weights, lambdaMax, ci, cr };
}

function mmult(A, v){
  const n = A.length;
  const out = Array(n).fill(0);
  for(let i=0;i<n;i++){
    let s=0;
    for(let j=0;j<n;j++) s += A[i][j]*v[j];
    out[i]=s;
  }
  return out;
}

function initMatrices(){
  state.criteriaMatrix = identityMatrix(state.criteria.length);
  state.altMatrices = state.criteria.map(()=> identityMatrix(state.alternatives.length));
}

initMatrices();

// ---------- UI helpers ----------
const elSteps = document.getElementById("steps");
const elStatus = document.getElementById("status");
const view1 = document.getElementById("view1");
const view2 = document.getElementById("view2");
const view3 = document.getElementById("view3");

function setStatus(msg){ elStatus.textContent = msg || ""; }

function renderSteps(){
  elSteps.innerHTML = "";
  const labels = ["Problem", "Matrices", "Results"];
  labels.forEach((t, i)=>{
    const d = document.createElement("div");
    d.className = "stepPill" + (state.step === i+1 ? " active" : "");
    d.textContent = `${i+1}. ${t}`;
    elSteps.appendChild(d);
  });
}

function showStep(n){
  state.step = n;
  view1.classList.toggle("hidden", n !== 1);
  view2.classList.toggle("hidden", n !== 2);
  view3.classList.toggle("hidden", n !== 3);
  renderSteps();
  render();
}

document.querySelectorAll(".navBtn").forEach(btn=>{
  btn.addEventListener("click", ()=> showStep(Number(btn.dataset.step)));
});

document.getElementById("btnReset").addEventListener("click", ()=>{
  state.problem = { name: "Logistics", goal: "Select the best warehouse location" };
  state.criteria = ["Transportation cost", "Delivery lead time", "Service reliability"];
  state.alternatives = ["Location A", "Location B", "Location C"];
  initMatrices();
  showStep(1);
  setStatus("Reset done.");
});

document.getElementById("btnExport").addEventListener("click", ()=>{
  const payload = computeResults();
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

// ---------- Rendering: Step 1 ----------
function renderProblem(){
  view1.innerHTML = `
    <div class="row">
      <div>
        <div class="panelTitle">Problem</div>
        <label class="small muted">Name</label>
        <input id="p_name" type="text" value="${escapeHtml(state.problem.name)}" />
        <div style="height:10px"></div>
        <label class="small muted">Goal</label>
        <input id="p_goal" type="text" value="${escapeHtml(state.problem.goal)}" />
      </div>

      <div>
        <div class="panelTitle">Scale</div>
        <div class="badge">1 equal, 3 moderate, 5 strong, 7 major, 9 extreme</div>
        <div style="height:12px"></div>
        <div class="small muted">Results show CR (Consistency Ratio). Target CR ≤ 0.10.</div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="row">
      <div>
        <div class="panelTitle">Criteria</div>
        <div id="crit_list"></div>
        <button class="btn" id="crit_add">Add criterion</button>
      </div>

      <div>
        <div class="panelTitle">Alternatives</div>
        <div id="alt_list"></div>
        <button class="btn" id="alt_add">Add alternative</button>
      </div>
    </div>
  `;

  const pName = document.getElementById("p_name");
  const pGoal = document.getElementById("p_goal");
  pName.addEventListener("input", e => state.problem.name = e.target.value);
  pGoal.addEventListener("input", e => state.problem.goal = e.target.value);

  renderEditableList("crit_list", state.criteria, (arr)=> {
    state.criteria = arr;
    initMatrices();
    render();
  }, 2);

  renderEditableList("alt_list", state.alternatives, (arr)=> {
    state.alternatives = arr;
    initMatrices();
    render();
  }, 2);

  document.getElementById("crit_add").addEventListener("click", ()=>{
    state.criteria.push(`C${state.criteria.length+1}`);
    initMatrices();
    render();
  });

  document.getElementById("alt_add").addEventListener("click", ()=>{
    state.alternatives.push(`A${state.alternatives.length+1}`);
    initMatrices();
    render();
  });
}

function renderEditableList(containerId, items, onChange, minLen){
  const root = document.getElementById(containerId);
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

function escapeHtml(s){
  return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
}

// ---------- Rendering: Step 2 ----------
function renderMatrices(){
  const critSolve = ahpSolve(state.criteriaMatrix);
  const crBadge = crToBadge(critSolve.cr);

  view2.innerHTML = `
    <div class="row">
      <div>
        <div class="panelTitle">Criteria comparisons</div>
        <div class="small muted">Pairwise sliders update the matrix. ${crBadge}</div>
        <div style="height:10px"></div>
        <div id="crit_pairs"></div>
      </div>
      <div>
        <div class="panelTitle">Criteria matrix</div>
        <div id="crit_matrix"></div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="panelTitle">Alternative comparisons by criterion</div>
    <div id="alt_tabs" style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom: 10px;"></div>
    <div class="row">
      <div>
        <div id="alt_pairs"></div>
      </div>
      <div>
        <div id="alt_matrix"></div>
      </div>
    </div>
  `;

  renderPairwiseBlock("crit_pairs", state.criteria, state.criteriaMatrix, (A)=> {
    state.criteriaMatrix = A;
    renderMatrices();
  });

  document.getElementById("crit_matrix").innerHTML = matrixTable(state.criteria, state.criteriaMatrix);

  // Tabs for criteria
  const tabs = document.getElementById("alt_tabs");
  let activeIdx = state._activeCritIdx ?? 0;
  if(activeIdx >= state.criteria.length) activeIdx = 0;
  state._activeCritIdx = activeIdx;

  tabs.innerHTML = "";
  state.criteria.forEach((c, i)=>{
    const b = document.createElement("button");
    b.className = "btn";
    b.style.width = "auto";
    b.textContent = c;
    b.style.opacity = i === activeIdx ? "1" : "0.75";
    b.addEventListener("click", ()=>{
      state._activeCritIdx = i;
      renderMatrices();
    });
    tabs.appendChild(b);
  });

  const A = state.altMatrices[activeIdx];
  const solve = ahpSolve(A);

  document.getElementById("alt_pairs").innerHTML =
    `<div class="small muted">${state.criteria[activeIdx]} comparisons. ${crToBadge(solve.cr)}</div><div style="height:10px"></div>` +
    pairwiseUI(state.alternatives, A, (B)=> {
      state.altMatrices[activeIdx] = B;
      renderMatrices();
    });

  document.getElementById("alt_matrix").innerHTML = matrixTable(state.alternatives, A);
}

function crToBadge(cr){
  const ok = cr <= 0.10;
  const cls = ok ? "badge good" : "badge warn";
  const label = ok ? "CR OK" : "CR high";
  return `<span class="${cls}">${label}: ${cr.toFixed(3)}</span>`;
}

function renderPairwiseBlock(containerId, labels, A, onUpdate){
  const root = document.getElementById(containerId);
  root.innerHTML = pairwiseUI(labels, A, onUpdate);
}

function pairwiseUI(labels, A, onUpdate){
  const n = labels.length;
  let html = "";
  for(let i=0;i<n;i++){
    for(let j=i+1;j<n;j++){
      const aij = A[i][j];
      // represent value as 1..9 and a preferRight boolean
      let preferRight = false;
      let v = aij;
      if(v < 1){
        preferRight = true;
        v = 1 / v;
      }
      v = Math.min(9, Math.max(1, Math.round(v)));

      html += `
        <div class="pairRow" data-i="${i}" data-j="${j}">
          <div>${escapeHtml(labels[i])}</div>
          <div class="pairMid">
            <label><input type="checkbox" class="pref" ${preferRight ? "checked" : ""}/> prefer right</label>
            <input type="range" class="rng" min="1" max="9" step="1" value="${v}" />
            <div style="width:22px; text-align:right">${v}</div>
          </div>
          <div style="text-align:right">${escapeHtml(labels[j])}</div>
        </div>
      `;
    }
  }

  // attach events after insert
  setTimeout(()=>{
    document.querySelectorAll(".pairRow").forEach(row=>{
      const i = Number(row.dataset.i);
      const j = Number(row.dataset.j);
      const rng = row.querySelector(".rng");
      const pref = row.querySelector(".pref");
      const valBox = row.querySelector(".pairMid div:last-child");

      const apply = ()=>{
        const raw = Number(rng.value);
        valBox.textContent = String(raw);
        const preferRight = pref.checked;
        const v = preferRight ? 1/raw : raw;
        const B = setPairwise(A, i, j, v);
        onUpdate(B);
      };

      rng.addEventListener("input", apply);
      pref.addEventListener("change", apply);
    });
  }, 0);

  return html;
}

function matrixTable(labels, A){
  const n = labels.length;
  let html = `<table><thead><tr><th></th>`;
  for(let j=0;j<n;j++) html += `<th>${escapeHtml(labels[j])}</th>`;
  html += `</tr></thead><tbody>`;
  for(let i=0;i<n;i++){
    html += `<tr><td>${escapeHtml(labels[i])}</td>`;
    for(let j=0;j<n;j++){
      html += `<td>${Number(A[i][j]).toFixed(3)}</td>`;
    }
    html += `</tr>`;
  }
  html += `</tbody></table>`;
  return html;
}

// ---------- Rendering: Step 3 ----------
function computeResults(){
  const crit = ahpSolve(state.criteriaMatrix);

  const altSolves = state.altMatrices.map(A => ahpSolve(A));
  const W_alt = state.altMatrices.map((_, idx) => altSolves[idx].weights); // per criterion, weights over alternatives

  // final scores: sum_j (wCrit[j] * wAlt_j[i])
  const m = state.alternatives.length;
  const n = state.criteria.length;
  const scores = Array(m).fill(0);
  for(let i=0;i<m;i++){
    let s=0;
    for(let j=0;j<n;j++){
      s += crit.weights[j] * W_alt[j][i];
    }
    scores[i]=s;
  }

  const ranking = state.alternatives
    .map((name, i)=>({ name, score: scores[i] }))
    .sort((a,b)=>b.score-a.score);

  return {
    problem: state.problem,
    criteria: state.criteria,
    alternatives: state.alternatives,
    criteriaMatrix: state.criteriaMatrix,
    altMatrices: state.altMatrices,
    results: {
      criteriaWeights: crit.weights,
      criteriaCR: crit.cr,
      altWeightsByCriterion: state.criteria.reduce((acc, c, i)=>{
        acc[c] = W_alt[i];
        return acc;
      }, {}),
      altCRByCriterion: state.criteria.reduce((acc, c, i)=>{
        acc[c] = altSolves[i].cr;
        return acc;
      }, {}),
      finalScores: scores,
      ranking
    }
  };
}

function renderResults(){
  const res = computeResults();

  const critRows = res.criteria.map((c, i)=>({ name: c, value: res.results.criteriaWeights[i] }));
  const scoreRows = res.alternatives.map((a, i)=>({ name: a, value: res.results.finalScores[i] }));

  const best = res.results.ranking[0];

  view3.innerHTML = `
    <div class="row">
      <div>
        <div class="panelTitle">Summary</div>
        <div style="font-size:18px; font-weight:700; margin-top:6px">${escapeHtml(best.name)}</div>
        <div class="small muted">Top ranked alternative</div>
        <div style="height:10px"></div>
        ${crToBadge(res.results.criteriaCR)}
        <div style="height:10px"></div>
        <div class="small muted">Final score: ${best.score.toFixed(4)}</div>
        <div class="divider"></div>
        <div class="panelTitle">Ranking</div>
        ${rankingTable(res.results.ranking)}
      </div>

      <div>
        <div class="panelTitle">Charts</div>
        <div class="small muted">Criteria weights and final scores.</div>
        <div style="height:10px"></div>
        <canvas id="chartCrit" width="900" height="320"></canvas>
        <div style="height:14px"></div>
        <canvas id="chartScore" width="900" height="320"></canvas>
      </div>
    </div>

    <div class="divider"></div>

    <div class="panelTitle">Alternative weights by criterion</div>
    ${altWeightsTables(res)}
  `;

  drawBarChart("chartCrit", "Criteria weights", critRows);
  drawBarChart("chartScore", "Final scores", scoreRows);
}

function rankingTable(items){
  let html = `<table><thead><tr><th>Alternative</th><th>Score</th></tr></thead><tbody>`;
  items.forEach(x=>{
    html += `<tr><td>${escapeHtml(x.name)}</td><td>${x.score.toFixed(6)}</td></tr>`;
  });
  html += `</tbody></table>`;
  return html;
}

function altWeightsTables(res){
  let html = `<div class="row">`;
  res.criteria.forEach((c)=>{
    const w = res.results.altWeightsByCriterion[c];
    const cr = res.results.altCRByCriterion[c];
    html += `
      <div>
        <div class="small muted">${escapeHtml(c)} ${crToBadge(cr)}</div>
        <div style="height:8px"></div>
        <table>
          <thead><tr><th>Alternative</th><th>Weight</th></tr></thead>
          <tbody>
            ${res.alternatives.map((a,i)=>`<tr><td>${escapeHtml(a)}</td><td>${w[i].toFixed(6)}</td></tr>`).join("")}
          </tbody>
        </table>
      </div>
    `;
  });
  html += `</div>`;
  return html;
}

// Simple canvas bar chart
function drawBarChart(canvasId, title, items){
  const c = document.getElementById(canvasId);
  const ctx = c.getContext("2d");

  const W = c.width;
  const H = c.height;

  ctx.clearRect(0,0,W,H);

  // title
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  ctx.font = "18px system-ui";
  ctx.fillText(title, 18, 28);

  const maxV = Math.max(...items.map(x=>x.value), 0.00001);
  const left = 18;
  const right = 18;
  const top = 44;
  const bottom = 18;

  const chartW = W - left - right;
  const chartH = H - top - bottom;

  // axes baseline
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
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

    // bar
    ctx.fillStyle = "rgba(255,255,255,0.20)";
    ctx.fillRect(bx, by, barW, h);

    // label
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.font = "12px system-ui";
    ctx.fillText(x.name, bx, top + chartH + 14);

    // value
    ctx.fillStyle = "rgba(255,255,255,0.90)";
    ctx.font = "12px system-ui";
    ctx.fillText(x.value.toFixed(3), bx, by - 6);
  });
}

// ---------- Main render ----------
function render(){
  setStatus(`Criteria: ${state.criteria.length}, Alternatives: ${state.alternatives.length}`);
  if(state.step === 1) renderProblem();
  if(state.step === 2) renderMatrices();
  if(state.step === 3) renderResults();
  renderSteps();
}

showStep(1);
