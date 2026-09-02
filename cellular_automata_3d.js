// Hyperflux 3D Cellular Automata
// Novel rule-set designed by GitHub Copilot, 2025

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const glCanvas = document.getElementById('glcanvas');
let gl = null;

// Simulation lattice dimensions (x,y,z)
let NX = 32, NY = 32, NZ = 32;
let N = NX * NY * NZ;            // cells per species
let voxelSize = 14; // for rendering, not sim spacing
let renderMode = 'iso';
// WebGL camera state
let camYaw = 0.6, camPitch = 0.5; // radians
let camDist = 75; // distance
let isOrbiting = false;
let lastMouse = {x:0,y:0};

// Multi-species: every field is one flat Float32Array of length numSpecies*N.
// Species s owns the contiguous plane [s*N, (s+1)*N) -- keeps the 26-neighbour
// gather inside one plane, and makes numSpecies===1 byte-identical to the old
// single-field behaviour.
let numSpecies = 1;
let boundary = 'wrap';   // wrap | reflect | absorb
let seedKind = 'random';
let paintSpecies = 0;

// core fields: four ping-pong buffers, rotated (never copied) each step
let grid, prev, echo, next;

// state
let isRunning = false;
let generation = 0;
let speed = 10;
let rafId = null;
let lastTime = 0;

// parameters
let params = {
  baseDecay: 0.88,
  echoWeight: 0.35,
  diffXY: 0.20,
  diffZ: 0.12,
  resBoost: 1.20,
  spin: 0.15,
  birthTh: 85,
  activeTh: 18,
  minEnergy: 8,
  inhibit: 0.60
};
// Transparency band selection
let transparentBand = 'none';

const idx = (x, y, z) => (z * NY + y) * NX + x;

function allocFields() {
  N = NX * NY * NZ;
  const n = numSpecies * N;
  grid = new Float32Array(n);
  prev = new Float32Array(n);
  echo = new Float32Array(n);
  next = new Float32Array(n);
  buildBoundaryTables();
  generation = 0;
  genMs = 0;        // the old timing says nothing about the new lattice
  if (backend === 'gpu' && gpu.ok) gpuAllocate();
}

// Boundary lookup tables: index with (i+1) for i in [-1..L].
// Value is the resolved index, or -1 meaning "outside the lattice".
// ponytail: tables only cover a +/-1 stencil, which is all this rule reads.
// A wider kernel (Lenia, marching-cubes normals) would need L+2r entries.
let BX, BY, BZ;
function buildAxisTable(L) {
  const t = new Int32Array(L + 2);
  for (let i = -1; i <= L; i++) {
    let v;
    if (i >= 0 && i < L) v = i;
    else if (boundary === 'wrap') v = (i + L) % L;
    else if (boundary === 'reflect') v = i < 0 ? 0 : L - 1; // zero-flux mirror
    else v = -1;                                            // absorb: void outside
    t[i + 1] = v;
  }
  return t;
}
function buildBoundaryTables() {
  BX = buildAxisTable(NX);
  BY = buildAxisTable(NY);
  BZ = buildAxisTable(NZ);
}

// Row addressing is hoisted out of the x loop: for a given (z,y) the nine
// (z+dz, y+dy) rows are resolved once, so the inner loop is pure array loads
// with no multiplies and no per-neighbour boundary math.
// rowBases[(dz+1)*3 + (dy+1)] = cell index of that row's x=0, or -1 if the row
// lies outside an absorbing boundary. Species offset is added by the caller.
const rowBases = new Int32Array(9);
function computeRowBases(z, y) {
  const z0 = BZ[z], z2 = BZ[z + 2], y0 = BY[y], y2 = BY[y + 2];
  rowBases[0] = (z0 < 0 || y0 < 0) ? -1 : (z0 * NY + y0) * NX;
  rowBases[1] = (z0 < 0)           ? -1 : (z0 * NY + y ) * NX;
  rowBases[2] = (z0 < 0 || y2 < 0) ? -1 : (z0 * NY + y2) * NX;
  rowBases[3] = (y0 < 0)           ? -1 : (z  * NY + y0) * NX;
  rowBases[4] =                           (z  * NY + y ) * NX;
  rowBases[5] = (y2 < 0)           ? -1 : (z  * NY + y2) * NX;
  rowBases[6] = (z2 < 0 || y0 < 0) ? -1 : (z2 * NY + y0) * NX;
  rowBases[7] = (z2 < 0)           ? -1 : (z2 * NY + y ) * NX;
  rowBases[8] = (z2 < 0 || y2 < 0) ? -1 : (z2 * NY + y2) * NX;
}

// Moore-26 gather for species plane `base`, using the rowBases already computed
// for this (z,y). x0/x2 are the resolved x-1 / x+1 columns (-1 = outside).
// Results land in scratch globals: 98k result objects per step was pure GC churn.
let nbActive = 0, nbTotal = 0;
function gather(base, x, x0, x2) {
  let active = 0, total = 0;
  const th = params.activeTh;
  for (let r = 0; r < 9; r++) {
    const rb = rowBases[r];
    if (rb < 0) continue;
    const b = base + rb;
    let e;
    if (x0 >= 0) { e = grid[b + x0]; total += e; if (e > th) active++; }
    if (r !== 4) { e = grid[b + x ]; total += e; if (e > th) active++; } // r===4 is the cell itself
    if (x2 >= 0) { e = grid[b + x2]; total += e; if (e > th) active++; }
  }
  nbActive = active; nbTotal = total;
}

// Unpublished Hyperflux Rules (3D):
// 1) Temporal Echo Coupling: energy blends with echo (t-2) using echoWeight.
// 2) Anisotropic Diffusion: stronger in XY than Z.
// 3) Parity-Driven Vortex Spin: energy shifts along (dx,dy) based on cell parity and neighbor imbalance.
// 4) Plane Resonance: if alignment across Z planes is near, amplify.
// 5) Boundary: wrap / reflect / absorb; birth requires an energy signature and active count band.
// 6) Species Inhibition: rival species suppress growth and block birth, producing territories.
function step() {
  const { baseDecay, echoWeight, diffXY, diffZ, resBoost, spin,
          birthTh, activeTh, minEnergy, inhibit } = params;
  const echoDenom = 1 + echoWeight;

  for (let s = 0; s < numSpecies; s++) {
    const base = s * N;
    for (let z = 0; z < NZ; z++) {
      for (let y = 0; y < NY; y++) {
        computeRowBases(z, y);
        const rCenter = rowBases[4], rYm = rowBases[3], rYp = rowBases[5],
              rZm = rowBases[1], rZp = rowBases[7];
        for (let x = 0; x < NX; x++) {
          const x0 = BX[x], x2 = BX[x + 2];
          const i = rCenter + x;
          const E = grid[base + i];
          const Ee = echo[base + i];
          gather(base, x, x0, x2);
          const active = nbActive, total = nbTotal;

          // Base mix and decay
          const mix = (E + Ee * echoWeight) / echoDenom;
          let val = mix * baseDecay;

          // Anisotropic diffusion: pull towards average neighbors, different XY/Z weights
          const avgN = total / 26;
          const xm = x0 >= 0 ? grid[base + rCenter + x0] : 0;
          const xp = x2 >= 0 ? grid[base + rCenter + x2] : 0;
          const ym = rYm >= 0 ? grid[base + rYm + x] : 0;
          const yp = rYp >= 0 ? grid[base + rYp + x] : 0;
          const zm = rZm >= 0 ? grid[base + rZm + x] : 0;
          const zp = rZp >= 0 ? grid[base + rZp + x] : 0;
          const neighborXY = (xm + xp + ym + yp) / 4;
          const neighborZ = (zm + zp) / 2;
          val += (neighborXY - E) * diffXY + (neighborZ - E) * diffZ;

          // Parity-driven vortex spin in XY:
          // Compute local gradient bias and rotate energy around the cell depending on (x+y+z)%2
          const parity = (x + y + z) & 1;
          const gx = neighborXY - xm;
          const gy = neighborXY - ym;
          // spin direction: parity decides clockwise/counter-clockwise contribution
          const spinDir = parity ? 1 : -1;
          val += spin * spinDir * (gx - gy);

          // Inter-species inhibition: rivals bleed this species' energy away.
          // Rival pressure = their energy here, plus their 6-face average.
          let rival = 0;
          if (numSpecies > 1) {
            for (let t = 0; t < numSpecies; t++) {
              if (t === s) continue;
              const tb = t * N;
              const faces = (
                (x0 >= 0 ? grid[tb + rCenter + x0] : 0) + (x2 >= 0 ? grid[tb + rCenter + x2] : 0) +
                (rYm >= 0 ? grid[tb + rYm + x] : 0) + (rYp >= 0 ? grid[tb + rYp + x] : 0) +
                (rZm >= 0 ? grid[tb + rZm + x] : 0) + (rZp >= 0 ? grid[tb + rZp + x] : 0)
              ) / 6;
              rival += grid[tb + i] * 0.5 + faces * 0.5;
            }
            val -= inhibit * rival;
          }

          // Birth and survival bands using energy signature
          // Energy signature based on neighbor sum mod segmented window, different bands for birth
          let birthBoost = 0;
          const signature = total % 120; // 3D band signature
          if (E <= minEnergy) {
            // birth if active in [3..6] or [10..12], signature hits narrow windows,
            // and no rival already holds the cell
            if (rival < activeTh &&
                ((active >= 3 && active <= 6 && signature > 50 && signature < 66) ||
                 (active >= 10 && active <= 12 && signature > 95 && signature < 108))) {
              birthBoost = birthTh + (signature % 20);
            }
          } else {
            // survival amplification in balanced neighborhoods
            if (active === 4 || active === 5 || active === 7) {
              val += (avgN - E) * 0.15;
            }
          }

          // Plane resonance: if energies across z-1, z, z+1 are aligned within small delta, boost
          if (Math.abs(E - zm) < 14 && Math.abs(E - zp) < 14 && E > 30) {
            val *= resBoost;
          }

          // Apply birth boost if triggered
          if (birthBoost > 0) val = Math.max(val, birthBoost);

          // Clamp and threshold
          val = Math.max(0, Math.min(255, val));
          next[base + i] = val > minEnergy ? val : 0;
        }
      }
    }
  }

  // Temporal memory update (echo = t-2). Buffers rotate; the stale echo buffer
  // becomes the next scratch target, so a step allocates nothing.
  const spare = echo;
  echo = prev; prev = grid; grid = next; next = spare;
  generation++;
}

// Rendering
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (renderMode === 'iso') {
    glCanvas.style.display = 'none';
    canvas.style.display = 'block';
    drawIsometric();
  }
  else if (renderMode === 'slice') {
    glCanvas.style.display = 'none';
    canvas.style.display = 'block';
    drawSlice();
  }
  else if (renderMode === 'depth') {
    glCanvas.style.display = 'none';
    canvas.style.display = 'block';
    drawDepthFog();
  }
  else if (renderMode === 'webgl3d') {
    drawWebGL3D();
  }
  updateStats();
}

// Per-species base hues, used only when numSpecies > 1.
const SPECIES_RGB = [[40, 170, 255], [255, 70, 170], [110, 255, 120]];

// Dominant species at flat cell index i -> scratch globals (no per-voxel alloc).
let domS = 0, domE = 0;
function dominantAt(i) {
  domS = 0; domE = grid[i];
  for (let s = 1; s < numSpecies; s++) {
    const e = grid[s * N + i];
    if (e > domE) { domE = e; domS = s; }
  }
}

function energyColor(e, z = 0, s = 0) {
  if (e <= 0) return null;
  if (isTransparent(e)) return null;
  // Map energy to color; optional z affects fogging in depth mode
  let r, g, b;
  if (numSpecies > 1) {
    // Hue is fixed per species; energy drives brightness, whitening at saturation.
    const bc = SPECIES_RGB[s % SPECIES_RGB.length];
    const t = Math.min(1, e / 255);
    const k = 0.28 + 0.72 * t;
    const w = Math.max(0, (t - 0.82) / 0.18);
    r = Math.round(bc[0] * k + (255 - bc[0] * k) * w);
    g = Math.round(bc[1] * k + (255 - bc[1] * k) * w);
    b = Math.round(bc[2] * k + (255 - bc[2] * k) * w);
  } else if (e < 50) { // deep-blue to cyan
    const t = e / 50;
    r = 0; g = Math.floor(120 + t * 120); b = 220;
  } else if (e < 120) { // cyan to yellow
    const t = (e - 50) / 70;
    r = Math.floor(t * 255); g = 255; b = Math.floor(220 - t * 180);
  } else if (e < 200) { // yellow to red
    const t = (e - 120) / 80;
    r = 255; g = Math.floor(255 - t * 255); b = 40;
  } else { // red to white
    const t = (e - 200) / 55;
    r = 255; g = Math.floor(t * 255); b = Math.floor(100 + t * 155);
  }
  if (renderMode === 'depth') {
    const fog = 1 - z / (NZ - 1);
    r = Math.floor(r * (0.6 + 0.4 * fog));
    g = Math.floor(g * (0.6 + 0.4 * fog));
    b = Math.floor(b * (0.6 + 0.4 * fog));
  }
  return `rgb(${r},${g},${b})`;
}
function isTransparent(e){
  switch(transparentBand){
    case 'low': return e>0 && e<50;
    case 'mid': return e>=50 && e<120;
    case 'high': return e>=120 && e<200;
    case 'max': return e>=200;
    default: return false;
  }
}

// Isometric voxel projection (simple diamond projection).
// The diamond is (NX+NY) wide and (NX+NY)/2 + NZ*0.6 tall in voxel units, so
// the origin is derived from that rather than pinned, or the lattice hangs off
// the bottom of the canvas at larger voxel sizes.
function isoOriginY() {
  const spanY = (NX + NY) * (voxelSize * 0.5) + NZ * (voxelSize * 0.6);
  return Math.max(voxelSize, (canvas.height - spanY) / 2);
}
function drawIsometric() {
  const s = voxelSize;
  const originX = canvas.width / 2;
  const originY = isoOriginY();
  for (let z = 0; z < NZ; z++) {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        dominantAt(idx(x, y, z));
        const col = energyColor(domE, z, domS);
        if (!col) continue;
        const isoX = originX + (x - y) * s;
        const isoY = originY + (x + y) * (s * 0.5) + z * (s * 0.6);
        const w = s - 2;
        ctx.fillStyle = col;
        ctx.fillRect(isoX - w / 2, isoY - w / 2, w, w);
      }
    }
  }
}

// Slice plane addressing: (u,v) are the two free axes, k the fixed one.
let sliceAxis = 'xy';
let sliceIndex = Math.floor(NZ / 2);

function sliceDepth() {
  return sliceAxis === 'xy' ? NZ : sliceAxis === 'xz' ? NY : NX;
}
function sliceUV() {
  if (sliceAxis === 'xy') return [NX, NY];
  if (sliceAxis === 'xz') return [NX, NZ];
  return [NY, NZ]; // yz
}
function sliceIdx(u, v, k) {
  if (sliceAxis === 'xy') return idx(u, v, k);
  if (sliceAxis === 'xz') return idx(u, k, v);
  return idx(k, u, v); // yz
}

function drawSlice() {
  const s = voxelSize;
  const startX = 20, startY = 60;
  const [uN, vN] = sliceUV();
  const k = Math.max(0, Math.min(sliceIndex, sliceDepth() - 1));
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  for (let v = 0; v < vN; v++) {
    for (let u = 0; u < uN; u++) {
      dominantAt(sliceIdx(u, v, k));
      const col = energyColor(domE, k, domS);
      if (!col) continue;
      ctx.fillStyle = col;
      ctx.fillRect(startX + u * s, startY + v * s, s - 2, s - 2);
    }
  }
  ctx.strokeRect(startX - 2, startY - 2, uN * s + 4, vN * s + 4);
  ctx.fillStyle = 'rgba(255,255,255,0.75)';
  ctx.font = '13px sans-serif';
  ctx.fillText(`${sliceAxis.toUpperCase()} plane @ ${k} / ${sliceDepth() - 1}`, startX, startY - 12);
}

function drawDepthFog() {
  const s = voxelSize;
  // render back to front by z
  const startX = 40, startY = 40;
  for (let z = NZ - 1; z >= 0; z--) {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        dominantAt(idx(x, y, z));
        const col = energyColor(domE, z, domS);
        if (!col) continue;
        const px = startX + x * (s - 4) + Math.floor(z * 0.35 * s);
        const py = startY + y * (s - 4) + Math.floor(z * 0.35 * s);
        ctx.fillStyle = col;
        ctx.fillRect(px, py, s - 3, s - 3);
      }
    }
  }
}

let lastLegendHTML = null;
function updateStats() {
  let active = 0, total = 0;
  const perSpecies = new Array(numSpecies).fill(0);
  for (let i = 0; i < N; i++) {
    dominantAt(i);
    if (domE > 0) { active++; total += domE; perSpecies[domS]++; }
  }
  const set = (id, v) => { document.getElementById(id).textContent = v; };
  set('roGen', generation.toLocaleString());
  set('roLive', active.toLocaleString());
  set('roAvg', active ? (total / active).toFixed(1) : '0.0');
  set('roSize', (NX === NY && NY === NZ) ? NX + '³' : `${NX}·${NY}·${NZ}`);
  set('roMs', genMs ? genMs.toFixed(1) : '–');

  // Per-species tally, only when there is competition to report.
  const html = numSpecies > 1
    ? perSpecies.map((c, s) =>
        `<div><span class="sw" style="background:rgb(${SPECIES_RGB[s].join(',')})"></span>` +
        `Species ${s + 1}<b>${c.toLocaleString()}</b></div>`).join('')
    : '';
  if (html !== lastLegendHTML) {
    document.getElementById('speciesLegend').innerHTML = html;
    lastLegendHTML = html;
  }
  // Show hint if in WebGL mode and nothing visible
  if (renderMode === 'webgl3d' && active === 0) {
    document.getElementById('settingsError').textContent = 'No active voxels to render in 3D. Paint or Randomize.';
  } else if (document.getElementById('settingsError').textContent.startsWith('No active voxels')) {
    document.getElementById('settingsError').textContent = '';
  }
}

// Interaction: paint voxels with max energy, drag moves across XY on current depth slice
let isPainting = false;

canvas.addEventListener('mousedown', (e) => { isPainting = true; paintAt(e); });
canvas.addEventListener('mousemove', (e) => { if (isPainting) paintAt(e); });
canvas.addEventListener('mouseup', () => { isPainting = false; });
canvas.addEventListener('mouseleave', () => { isPainting = false; });

function paintAt(e) {
  // The canvas is CSS-scaled to fit the stage, so pointer coords have to be
  // mapped back into the backing store before they mean anything.
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  const ps = Math.min(paintSpecies, numSpecies - 1) * N;
  if (renderMode === 'slice') {
    const s = voxelSize;
    const startX = 20, startY = 60;
    const [uN, vN] = sliceUV();
    const gu = Math.floor((x - startX) / s);
    const gv = Math.floor((y - startY) / s);
    if (gu >= 0 && gu < uN && gv >= 0 && gv < vN) {
      const k = Math.max(0, Math.min(sliceIndex, sliceDepth() - 1));
      grid[ps + sliceIdx(gu, gv, k)] = 255;
      gpuDirty = true;
      draw();
    }
  } else {
    // in iso/depth modes, approximate XY by inverse of projection near the middle region
    // This is a simple heuristic to keep interaction usable.
    const s = voxelSize;
    const originX = canvas.width / 2;
    const originY = isoOriginY();
    // try multiple z layers near the working depth to find closest pixel
    const paintZ = sliceAxis === 'xy'
      ? Math.max(0, Math.min(sliceIndex, NZ - 1))
      : Math.floor(NZ / 2);
    let best = null;
    for (let z = Math.max(0, paintZ - 2); z <= Math.min(NZ - 1, paintZ + 2); z++) {
      for (let yy = 0; yy < NY; yy += 2) {
        for (let xx = 0; xx < NX; xx += 2) {
          const isoX = originX + (xx - yy) * s;
          const isoY = originY + (xx + yy) * (s * 0.5) + z * (s * 0.6);
          const dx = isoX - x, dy = isoY - y;
          const d2 = dx * dx + dy * dy;
          if (!best || d2 < best.d2) best = { x: xx, y: yy, z, d2 };
        }
      }
    }
    if (best) {
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const xx = Math.min(NX - 1, best.x + dx);
        const yy = Math.min(NY - 1, best.y + dy);
        grid[ps + idx(xx, yy, best.z)] = 255;
      }
      gpuDirty = true;
      draw();
    }
  }
}

// Controls
const runBtn = document.getElementById('runBtn');
function setRunning(on) {
  isRunning = on;
  runBtn.textContent = on ? 'Pause' : 'Run';
  document.getElementById('liveDot').classList.toggle('on', on);
  if (on) {
    lastTime = performance.now();
    rafId = requestAnimationFrame(loop);
  } else if (rafId) {
    cancelAnimationFrame(rafId);
  }
}
runBtn.addEventListener('click', () => setRunning(!isRunning));
document.getElementById('stepBtn').addEventListener('click', timedGeneration);
document.getElementById('clearBtn').addEventListener('click', () => { clearAll(); draw(); });

document.addEventListener('keydown', (e) => {
  // e.target is Document when nothing is focused, and Document has no closest()
  const t = e.target;
  if (t instanceof Element && t.closest('input, textarea, select')) return;
  if (e.code === 'Space') { e.preventDefault(); setRunning(!isRunning); }
  else if (e.key === 's' || e.key === 'S') { step(); draw(); }
});

// Fill the bar behind a parameter row so the track reads as its own value.
function setFill(el) {
  const min = parseFloat(el.min), max = parseFloat(el.max);
  const pct = max > min ? ((parseFloat(el.value) - min) / (max - min)) * 100 : 0;
  el.parentElement.style.setProperty('--fill', pct.toFixed(2) + '%');
}
function refreshFills() {
  document.querySelectorAll('.par input[type="range"]').forEach(setFill);
}

function setRenderMode(m) {
  renderMode = m;
  for (const b of document.querySelectorAll('#renderSeg button')) {
    b.setAttribute('aria-pressed', String(b.dataset.mode === m));
  }
  if (m === 'webgl3d') ensureWebGL();
}
for (const b of document.querySelectorAll('#renderSeg button')) {
  b.addEventListener('click', () => { setRenderMode(b.dataset.mode); draw(); refreshSettingsText(); });
}

function setTransparentBand(band) {
  transparentBand = band;
  for (const b of document.querySelectorAll('#ramp button')) {
    b.setAttribute('aria-pressed', String(b.dataset.band === band));
  }
}
for (const b of document.querySelectorAll('#ramp button')) {
  // clicking the active band clears it, so the ramp is its own off switch
  b.addEventListener('click', () => {
    setTransparentBand(transparentBand === b.dataset.band ? 'none' : b.dataset.band);
    draw();
    refreshSettingsText();
  });
}

// "Paint as" only means anything once a second species exists.
function syncSpeciesUI() {
  document.getElementById('paintField').style.display = numSpecies > 1 ? '' : 'none';
}

// Largest voxel size whose isometric diamond still fits the canvas. Without
// this an 80³ lattice draws well past the edges at any slider value.
function fitVoxelSize() {
  const byW = canvas.width / (NX + NY);
  const byH = canvas.height / ((NX + NY) * 0.5 + NZ * 0.6);
  return Math.max(3, Math.min(22, Math.floor(Math.min(byW, byH))));
}

function syncSizeUI() {
  const cube = (NX === NY && NY === NZ) ? NX : 0;
  for (const b of document.querySelectorAll('#sizeSeg button')) {
    b.setAttribute('aria-pressed', String(+b.dataset.size === cube));
  }
  const mb = (4 * numSpecies * N * 4) / 1048576;   // 4 buffers, 4 bytes a cell
  document.getElementById('sizeNote').textContent =
    `${N.toLocaleString()} cells · ${mb.toFixed(1)} MB`;
}

function setLattice(n) {
  NX = NY = NZ = n;
  allocFields();
  seed(seedKind);
  sliceIndex = Math.floor(n / 2);
  // scale the voxels to the new lattice so the view stays framed
  voxelSize = fitVoxelSize();
  const vs = document.getElementById('voxelSizeSlider');
  vs.value = voxelSize;
  document.getElementById('voxelSizeValue').textContent = voxelSize;
  setFill(vs);
  syncSliceUI();
  syncSizeUI();
}

for (const b of document.querySelectorAll('#sizeSeg button')) {
  b.addEventListener('click', () => { setLattice(+b.dataset.size); draw(); refreshSettingsText(); });
}

// Rolling cost of one generation, measured here rather than quoted, because
// it depends entirely on the machine this is running on.
let genMs = 0;
let inFlight = false;
async function timedGeneration() {
  if (inFlight) return;          // a GPU generation is still resolving
  inFlight = true;
  const t0 = performance.now();
  try {
    if (backend === 'gpu' && gpu.ok) await gpuStep();
    else step();
    draw();
  } finally {
    const dt = performance.now() - t0;
    genMs = genMs ? genMs * 0.8 + dt * 0.2 : dt;
    inFlight = false;
  }
}
document.getElementById('randomBtn').addEventListener('click', () => { seed(seedKind); draw(); });

document.getElementById('seedKind').addEventListener('change', (e) => {
  seedKind = e.target.value;
  seed(seedKind);
  draw();
  refreshSettingsText();
});
document.getElementById('boundarySelect').addEventListener('change', (e) => {
  boundary = e.target.value;
  buildBoundaryTables();
  draw();
  refreshSettingsText();
});
document.getElementById('speciesSelect').addEventListener('change', (e) => {
  numSpecies = parseInt(e.target.value, 10);
  allocFields();      // buffer length depends on species count
  seed(seedKind);
  syncSpeciesUI();
  syncSizeUI();       // memory figure scales with species
  draw();
  refreshSettingsText();
});
document.getElementById('paintSpeciesSelect').addEventListener('change', (e) => {
  paintSpecies = parseInt(e.target.value, 10);
});
document.getElementById('sliceAxisSelect').addEventListener('change', (e) => {
  sliceAxis = e.target.value;
  syncSliceUI();
  draw();
  refreshSettingsText();
});
document.getElementById('sliceIndexSlider').addEventListener('input', (e) => {
  sliceIndex = parseInt(e.target.value, 10);
  document.getElementById('sliceIndexValue').textContent = sliceIndex;
  setFill(e.target);
  draw();
  refreshSettingsText();
});

// Keep the slice slider's range matched to the current fixed axis.
function syncSliceUI() {
  const sl = document.getElementById('sliceIndexSlider');
  const max = sliceDepth() - 1;
  sl.max = max;
  if (sliceIndex > max) sliceIndex = max;
  sl.value = sliceIndex;
  setFill(sl);
  document.getElementById('sliceIndexValue').textContent = sliceIndex;
  document.getElementById('sliceAxisSelect').value = sliceAxis;
}

document.getElementById('speedSlider').addEventListener('input', (e) => {
  speed = parseInt(e.target.value, 10);
  document.getElementById('speedValue').textContent = speed;
  setFill(e.target);
  refreshSettingsText();
});
document.getElementById('voxelSizeSlider').addEventListener('input', (e) => {
  voxelSize = parseInt(e.target.value, 10);
  document.getElementById('voxelSizeValue').textContent = voxelSize;
  setFill(e.target);
  draw();
  refreshSettingsText();
});

const bind = (id, key, fmt = (v)=>v) => {
  const el = document.getElementById(id);
  const label = document.getElementById(id + 'Value');
  el.addEventListener('input', (e) => {
    const v = el.type === 'range' ? parseFloat(e.target.value) : e.target.value;
    params[key] = el.type === 'range' ? v : params[key];
    label.textContent = (typeof v === 'number') ? (el.step && parseFloat(el.step) < 1 ? v.toFixed(2) : v) : fmt(v);
    setFill(el);
    refreshSettingsText();
  });
};
bind('baseDecay', 'baseDecay');
bind('echoWeight', 'echoWeight');
bind('diffXY', 'diffXY');
bind('diffZ', 'diffZ');
bind('resBoost', 'resBoost');
bind('spin', 'spin');
bind('birthTh', 'birthTh');
bind('activeTh', 'activeTh');
bind('minEnergy', 'minEnergy');
bind('inhibit', 'inhibit');

// Presets and randomizer
function loadPreset(name) {
  let p;
  if (name === 'default') p = { baseDecay: 0.88, echoWeight: 0.35, diffXY: 0.20, diffZ: 0.12, resBoost: 1.20, spin: 0.15, birthTh: 85, activeTh: 18, minEnergy: 8, inhibit: 0.60 };
  else if (name === 'waves') p = { baseDecay: 0.92, echoWeight: 0.55, diffXY: 0.12, diffZ: 0.10, resBoost: 1.35, spin: 0.08, birthTh: 90, activeTh: 20, minEnergy: 6, inhibit: 0.45 };
  else if (name === 'vortex') p = { baseDecay: 0.84, echoWeight: 0.25, diffXY: 0.24, diffZ: 0.16, resBoost: 1.25, spin: 0.28, birthTh: 95, activeTh: 16, minEnergy: 8, inhibit: 0.70 };
  else if (name === 'chaos') p = { baseDecay: 0.78, echoWeight: 0.40, diffXY: 0.30, diffZ: 0.22, resBoost: 1.45, spin: 0.32, birthTh: 110, activeTh: 14, minEnergy: 5, inhibit: 0.95 };
  params = { ...params, ...p };
  syncParamUI();
}

function randomizeParams() {
  const rnd = (a, b) => a + Math.random() * (b - a);
  const p = {
    baseDecay: +rnd(0.72, 0.95).toFixed(2),
    echoWeight: +rnd(0.0, 0.8).toFixed(2),
    diffXY: +rnd(0.05, 0.35).toFixed(2),
    diffZ: +rnd(0.05, 0.30).toFixed(2),
    resBoost: +rnd(1.05, 1.7).toFixed(2),
    spin: +rnd(0.05, 0.35).toFixed(2),
    birthTh: Math.floor(rnd(60, 160)),
    activeTh: Math.floor(rnd(8, 40)),
    minEnergy: Math.floor(rnd(0, 20)),
    inhibit: +rnd(0.2, 1.2).toFixed(2)
  };
  if (p.diffZ > p.diffXY) p.diffZ = +(p.diffXY - 0.02).toFixed(2);
  params = { ...params, ...p };
  syncParamUI();
}

function syncParamUI() {
  const set = (id, v, fmt = (x)=>x) => {
    const el = document.getElementById(id);
    const label = document.getElementById(id + 'Value');
    el.value = v;
    label.textContent = (typeof v === 'number') ? ((el.step && parseFloat(el.step) < 1) ? (+v).toFixed(2) : v) : fmt(v);
    setFill(el);
  };
  set('baseDecay', params.baseDecay);
  set('echoWeight', params.echoWeight);
  set('diffXY', params.diffXY);
  set('diffZ', params.diffZ);
  set('resBoost', params.resBoost);
  set('spin', params.spin);
  set('birthTh', params.birthTh);
  set('activeTh', params.activeTh);
  set('minEnergy', params.minEnergy);
  set('inhibit', params.inhibit);
  refreshSettingsText();
}

// JSON settings serialization
function getSettingsObject() {
  return {
    NX, NY, NZ,
    numSpecies,
    boundary,
    seedKind,
    sliceAxis,
    sliceIndex,
    speed,
    voxelSize,
    renderMode,
    transparentBand,
    params: { ...params }
  };
}
function refreshSettingsText() {
  const ta = document.getElementById('settingsText');
  if (!ta) return;
  ta.value = JSON.stringify(getSettingsObject(), null, 2);
}
function applySettingsFromText() {
  const ta = document.getElementById('settingsText');
  const errEl = document.getElementById('settingsError');
  errEl.textContent = '';
  try {
    const obj = JSON.parse(ta.value);
    let needsRealloc = false;
    if (obj.NX && obj.NY && obj.NZ && (obj.NX !== NX || obj.NY !== NY || obj.NZ !== NZ)) {
      NX = obj.NX; NY = obj.NY; NZ = obj.NZ;
      needsRealloc = true;
    }
    if (obj.numSpecies && obj.numSpecies !== numSpecies) {
      numSpecies = Math.max(1, Math.min(3, obj.numSpecies));
      document.getElementById('speciesSelect').value = String(numSpecies);
      needsRealloc = true;
    }
    if (obj.boundary && obj.boundary !== boundary) {
      boundary = obj.boundary;
      document.getElementById('boundarySelect').value = boundary;
      buildBoundaryTables();
    }
    if (obj.seedKind) {
      seedKind = obj.seedKind;
      document.getElementById('seedKind').value = seedKind;
    }
    if (needsRealloc) { allocFields(); seed(seedKind); }
    if (obj.sliceAxis) sliceAxis = obj.sliceAxis;
    if (typeof obj.sliceIndex === 'number') sliceIndex = obj.sliceIndex;
    syncSliceUI();
    if (obj.params) {
      params = { ...params, ...obj.params };
      syncParamUI();
    } else {
      refreshSettingsText();
    }
    if (typeof obj.speed === 'number') {
      speed = obj.speed;
      const ss = document.getElementById('speedSlider');
      ss.value = speed; document.getElementById('speedValue').textContent = speed;
    }
    if (typeof obj.voxelSize === 'number') {
      voxelSize = obj.voxelSize;
      const vs = document.getElementById('voxelSizeSlider');
      vs.value = voxelSize; document.getElementById('voxelSizeValue').textContent = voxelSize;
    }
    if (obj.renderMode) setRenderMode(obj.renderMode);
    if (obj.transparentBand) setTransparentBand(obj.transparentBand);
    syncSpeciesUI();
    syncSizeUI();
    refreshFills();
    draw();
    refreshSettingsText();
  } catch (e) {
    errEl.textContent = 'Invalid JSON: ' + e.message;
  }
}

// Preset buttons
const presetBtns = document.querySelectorAll('[data-preset]');
for (const b of presetBtns) {
  b.addEventListener('click', () => {
    loadPreset(b.dataset.preset);
    for (const o of presetBtns) o.setAttribute('aria-pressed', String(o === b));
    draw();
  });
}
document.getElementById('randomizeAndShowBtn').addEventListener('click', () => {
  for (const o of presetBtns) o.setAttribute('aria-pressed', 'false');
});

// Settings buttons
document.getElementById('applySettingsBtn').addEventListener('click', applySettingsFromText);
document.getElementById('copySettingsBtn').addEventListener('click', () => {
  navigator.clipboard.writeText(document.getElementById('settingsText').value).catch(()=>{});
});
document.getElementById('refreshSettingsBtn').addEventListener('click', refreshSettingsText);
document.getElementById('randomizeAndShowBtn').addEventListener('click', () => { randomizeParams(); refreshSettingsText(); });

// Auto-parse on blur for convenience
document.getElementById('settingsText').addEventListener('blur', () => {
  if (document.getElementById('settingsError').textContent) return; // don't auto-apply if error shown
  applySettingsFromText();
});

// Simulation loop
function loop(ts) {
  if (!isRunning) return;
  const interval = 1000 / speed;
  if (ts - lastTime >= interval) {
    timedGeneration();
    lastTime = ts;
  }
  rafId = requestAnimationFrame(loop);
}

// Utilities
function clearAll() {
  grid.fill(0); prev.fill(0); echo.fill(0); next.fill(0);
  generation = 0;
  if (backend === 'gpu' && gpu.ok) gpuReset();
}

// Seed patterns. All write into `grid` only; echo/prev stay zero so the first
// couple of generations build their own temporal memory.
function seed(kind = 'random') {
  clearAll();
  const S = numSpecies;
  const put = (x, y, z, s) => {
    x = Math.round(x); y = Math.round(y); z = Math.round(z);
    if (x < 0 || y < 0 || z < 0 || x >= NX || y >= NY || z >= NZ) return;
    grid[s * N + idx(x, y, z)] = 60 + Math.random() * 195;
  };
  const cx = (NX - 1) / 2, cy = (NY - 1) / 2, cz = (NZ - 1) / 2;

  if (kind === 'sphere') {
    // Hollow shell; species split by azimuth so they start as wedges.
    const R = Math.min(NX, NY, NZ) * 0.35;
    for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
      if (Math.abs(Math.hypot(x - cx, y - cy, z - cz) - R) > 0.9) continue;
      const ang = Math.atan2(y - cy, x - cx) + Math.PI;
      put(x, y, z, Math.min(S - 1, Math.floor(ang / (2 * Math.PI) * S)));
    }
  } else if (kind === 'plane') {
    // One noisy mid-plane; energy has to spread along Z to fill the volume.
    const z = Math.floor(cz);
    for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
      if (Math.random() < 0.55) put(x, y, z, Math.floor(Math.random() * S));
    }
  } else if (kind === 'filament') {
    // One random walk per species, thickened so it survives the decay term.
    const steps = NX * 6;
    for (let s = 0; s < S; s++) {
      let x = Math.random() * NX, y = Math.random() * NY, z = Math.random() * NZ;
      for (let i = 0; i < steps; i++) {
        put(x, y, z, s); put(x + 1, y, z, s); put(x, y + 1, z, s);
        x = Math.max(0, Math.min(NX - 1, x + (Math.random() - 0.5) * 2.4));
        y = Math.max(0, Math.min(NY - 1, y + (Math.random() - 0.5) * 2.4));
        z = Math.max(0, Math.min(NZ - 1, z + (Math.random() - 0.5) * 2.4));
      }
    }
  } else if (kind === 'symmetric') {
    // Random points in one octant, mirrored to all eight -> mandala growth.
    // Mirrors keep the same species so the symmetry survives inhibition.
    const count = Math.floor(N * 0.006) + 20;
    for (let i = 0; i < count; i++) {
      const px = Math.floor(Math.random() * (NX / 2));
      const py = Math.floor(Math.random() * (NY / 2));
      const pz = Math.floor(Math.random() * (NZ / 2));
      const s = Math.floor(Math.random() * S);
      for (let mx = 0; mx < 2; mx++) for (let my = 0; my < 2; my++) for (let mz = 0; mz < 2; mz++) {
        put(mx ? NX - 1 - px : px, my ? NY - 1 - py : py, mz ? NZ - 1 - pz : pz, s);
      }
    }
  } else { // random
    for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
      if (Math.random() < 0.12) put(x, y, z, Math.floor(Math.random() * S));
    }
  }
  if (backend === 'gpu' && gpu.ok) gpuReset();
}

// ---- WebGPU backend -------------------------------------------------
// The compute shader below is a port of step(): same terms, same order, same
// boundary handling. The CPU array stays the source of truth -- every GPU
// generation is read back into `grid` -- so the renderers, stats, painting,
// JSON panel and self-test all keep working without knowing which backend ran.
// Trajectories will drift slightly from the CPU path: WGSL is f32 throughout
// while JS computes in f64 and rounds on store, and the rule is chaotic.
let backend = 'cpu';
let gpuDirty = false;                       // CPU painted since the last upload
const gpu = { ok:false, device:null, queue:null, pipe:null, uni:null,
              bufs:null, staging:null, groups:null, phase:0, bytes:0 };

// [grid, echo, next] buffer indices. The CPU rotation has period 4, so four
// bind groups cover every state and none is rebuilt per frame.
const PHASES = [[0,2,3],[3,1,2],[2,0,1],[1,3,0]];

const WGSL = `
struct P {
  nx:u32, ny:u32, nz:u32, numSpecies:u32,
  boundary:u32, pad0:u32, pad1:u32, pad2:u32,
  baseDecay:f32, echoWeight:f32, diffXY:f32, diffZ:f32,
  resBoost:f32, spin:f32, birthTh:f32, activeTh:f32,
  minEnergy:f32, inhibit:f32, pad3:f32, pad4:f32,
};
@group(0) @binding(0) var<storage, read>       G  : array<f32>;
@group(0) @binding(1) var<storage, read>       EC : array<f32>;
@group(0) @binding(2) var<storage, read_write> NX : array<f32>;
@group(0) @binding(3) var<uniform>             p  : P;

// Resolve one axis; -1 means outside an absorbing boundary.
fn res(v:i32, n:u32) -> i32 {
  let m = i32(n);
  if (v >= 0 && v < m) { return v; }
  if (p.boundary == 0u) { return ((v % m) + m) % m; }          // wrap
  if (p.boundary == 1u) { if (v < 0) { return 0; } return m - 1; }  // reflect
  return -1;                                                    // absorb
}
fn cell(base:u32, xi:i32, yi:i32, zi:i32) -> f32 {
  if (xi < 0 || yi < 0 || zi < 0) { return 0.0; }
  return G[base + (u32(zi) * p.ny + u32(yi)) * p.nx + u32(xi)];
}

@compute @workgroup_size(4,4,4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= p.nx || gid.y >= p.ny || gid.z >= p.nz) { return; }
  let x = i32(gid.x); let y = i32(gid.y); let z = i32(gid.z);
  let i = (gid.z * p.ny + gid.y) * p.nx + gid.x;
  let N = p.nx * p.ny * p.nz;

  let xm = res(x-1, p.nx); let xp = res(x+1, p.nx);
  let ym = res(y-1, p.ny); let yp = res(y+1, p.ny);
  let zm = res(z-1, p.nz); let zp = res(z+1, p.nz);

  for (var s:u32 = 0u; s < p.numSpecies; s = s + 1u) {
    let base = s * N;
    let E  = G[base + i];
    let Ee = EC[base + i];

    var total = 0.0;
    var nAct = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
      let zz = res(z+dz, p.nz);
      for (var dy = -1; dy <= 1; dy = dy + 1) {
        let yy = res(y+dy, p.ny);
        for (var dx = -1; dx <= 1; dx = dx + 1) {
          if (dx == 0 && dy == 0 && dz == 0) { continue; }
          let e = cell(base, res(x+dx, p.nx), yy, zz);
          total = total + e;
          if (e > p.activeTh) { nAct = nAct + 1.0; }
        }
      }
    }

    var val = ((E + Ee * p.echoWeight) / (1.0 + p.echoWeight)) * p.baseDecay;

    let avgN = total / 26.0;
    let vxm = cell(base, xm, y, z); let vxp = cell(base, xp, y, z);
    let vym = cell(base, x, ym, z); let vyp = cell(base, x, yp, z);
    let vzm = cell(base, x, y, zm); let vzp = cell(base, x, y, zp);
    let nXY = (vxm + vxp + vym + vyp) / 4.0;
    let nZ  = (vzm + vzp) / 2.0;
    val = val + (nXY - E) * p.diffXY + (nZ - E) * p.diffZ;

    var spinDir = -1.0;
    if (((gid.x + gid.y + gid.z) & 1u) == 1u) { spinDir = 1.0; }
    val = val + p.spin * spinDir * ((nXY - vxm) - (nXY - vym));

    var rival = 0.0;
    if (p.numSpecies > 1u) {
      for (var t:u32 = 0u; t < p.numSpecies; t = t + 1u) {
        if (t == s) { continue; }
        let tb = t * N;
        let faces = (cell(tb,xm,y,z) + cell(tb,xp,y,z) +
                     cell(tb,x,ym,z) + cell(tb,x,yp,z) +
                     cell(tb,x,y,zm) + cell(tb,x,y,zp)) / 6.0;
        rival = rival + G[tb + i] * 0.5 + faces * 0.5;
      }
      val = val - p.inhibit * rival;
    }

    var birth = 0.0;
    let sig = total - floor(total / 120.0) * 120.0;
    if (E <= p.minEnergy) {
      if (rival < p.activeTh &&
          ((nAct >= 3.0 && nAct <= 6.0 && sig > 50.0 && sig < 66.0) ||
           (nAct >= 10.0 && nAct <= 12.0 && sig > 95.0 && sig < 108.0))) {
        birth = p.birthTh + (sig - floor(sig / 20.0) * 20.0);
      }
    } else {
      if (nAct == 4.0 || nAct == 5.0 || nAct == 7.0) { val = val + (avgN - E) * 0.15; }
    }

    if (abs(E - vzm) < 14.0 && abs(E - vzp) < 14.0 && E > 30.0) { val = val * p.resBoost; }
    if (birth > 0.0) { val = max(val, birth); }
    val = clamp(val, 0.0, 255.0);
    if (val <= p.minEnergy) { val = 0.0; }
    NX[base + i] = val;
  }
}`;

const uniData = new ArrayBuffer(80);
const uniU32 = new Uint32Array(uniData, 0, 5);
const uniF32 = new Float32Array(uniData, 32, 10);

async function initGPU() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    const device = await adapter.requestDevice();
    const module = device.createShaderModule({ code: WGSL });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter(m => m.type === 'error');
    if (errs.length) {
      console.error('WGSL:', errs.map(m => `${m.lineNum}: ${m.message}`).join('\n'));
      return false;
    }
    gpu.device = device;
    gpu.queue = device.queue;
    gpu.pipe = device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: 'main' } });
    gpu.uni = device.createBuffer({ size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.lost.then(() => { gpu.ok = false; setBackend('cpu'); });
    gpu.ok = true;
    return true;
  } catch (e) {
    console.warn('WebGPU unavailable:', e.message);
    return false;
  }
}

function gpuAllocate() {
  if (!gpu.ok) return;
  if (gpu.bufs) for (const b of gpu.bufs) b.destroy();
  if (gpu.staging) gpu.staging.destroy();
  gpu.bytes = numSpecies * N * 4;
  gpu.bufs = [0,1,2,3].map(() => gpu.device.createBuffer({ size: gpu.bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }));
  gpu.staging = gpu.device.createBuffer({ size: gpu.bytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpu.groups = PHASES.map(([g, e, n]) => gpu.device.createBindGroup({
    layout: gpu.pipe.getBindGroupLayout(0),
    entries: [ { binding:0, resource:{ buffer: gpu.bufs[g] } },
               { binding:1, resource:{ buffer: gpu.bufs[e] } },
               { binding:2, resource:{ buffer: gpu.bufs[n] } },
               { binding:3, resource:{ buffer: gpu.uni } } ] }));
  gpuReset();
}

// Full restart: push the CPU lattice and clear the temporal history.
function gpuReset() {
  if (!gpu.ok || !gpu.bufs) return;
  gpu.phase = 0;
  const zeros = new Float32Array(numSpecies * N);
  gpu.queue.writeBuffer(gpu.bufs[PHASES[0][1]], 0, zeros);   // echo
  gpu.queue.writeBuffer(gpu.bufs[1], 0, zeros);              // prev
  gpu.queue.writeBuffer(gpu.bufs[PHASES[0][0]], 0, grid);
  gpuDirty = false;
}

// Push just the current lattice, keeping echo/prev intact (used after painting).
function gpuPushGrid() {
  if (!gpu.ok || !gpu.bufs) return;
  gpu.queue.writeBuffer(gpu.bufs[PHASES[gpu.phase][0]], 0, grid);
  gpuDirty = false;
}

function gpuUniforms() {
  uniU32.set([NX, NY, NZ, numSpecies,
    boundary === 'wrap' ? 0 : boundary === 'reflect' ? 1 : 2]);
  uniF32.set([params.baseDecay, params.echoWeight, params.diffXY, params.diffZ,
    params.resBoost, params.spin, params.birthTh, params.activeTh,
    params.minEnergy, params.inhibit]);
  gpu.queue.writeBuffer(gpu.uni, 0, uniData);
}

async function gpuStep() {
  if (gpuDirty) gpuPushGrid();
  gpuUniforms();                       // sliders can move between generations
  const enc = gpu.device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(gpu.pipe);
  pass.setBindGroup(0, gpu.groups[gpu.phase]);
  pass.dispatchWorkgroups(Math.ceil(NX/4), Math.ceil(NY/4), Math.ceil(NZ/4));
  pass.end();
  gpu.phase = (gpu.phase + 1) & 3;     // the buffer just written is the new grid
  enc.copyBufferToBuffer(gpu.bufs[PHASES[gpu.phase][0]], 0, gpu.staging, 0, gpu.bytes);
  gpu.queue.submit([enc.finish()]);
  await gpu.staging.mapAsync(GPUMapMode.READ, 0, gpu.bytes);
  grid.set(new Float32Array(gpu.staging.getMappedRange(0, gpu.bytes)));
  gpu.staging.unmap();
  generation++;
}

function setBackend(b) {
  if (b === 'gpu' && !gpu.ok) return;
  backend = b;
  if (b === 'gpu') gpuAllocate();
  for (const el of document.querySelectorAll('#backendSeg button')) {
    el.setAttribute('aria-pressed', String(el.dataset.backend === backend));
  }
  document.getElementById('roBackend').textContent = backend.toUpperCase();
  genMs = 0;                            // old timing describes the other backend
}

for (const b of document.querySelectorAll('#backendSeg button')) {
  b.addEventListener('click', () => { setBackend(b.dataset.backend); draw(); });
}

// Does the GPU port agree with the CPU rule? Runs with ?selftest=1 once a
// device exists; appended to the synchronous self-test results.
async function gpuSelfTest() {
  const saved = { NX, NY, NZ, numSpecies, boundary, backend };
  let verdict;
  try {
    NX = NY = NZ = 16; numSpecies = 2; boundary = 'wrap';
    backend = 'cpu';
    allocFields();
    for (let i = 0; i < grid.length; i++) {
      grid[i] = Math.random() < 0.3 ? 40 + Math.random() * 215 : 0;
    }
    const start = Float32Array.from(grid);

    step();                                   // CPU reference
    const cpuOut = Float32Array.from(grid);

    grid.set(start);                          // same input, GPU path
    prev.fill(0); echo.fill(0);
    backend = 'gpu';
    gpuAllocate();
    await gpuStep();

    let maxDiff = 0, live = 0;
    for (let i = 0; i < cpuOut.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(cpuOut[i] - grid[i]));
      if (cpuOut[i] > 0) live++;
    }
    verdict = (maxDiff < 0.05 && live > 0)
      ? `PASS gpu matches cpu (max diff ${maxDiff.toExponential(1)} over ${live} live cells)`
      : `FAIL gpu differs from cpu (max diff ${maxDiff}, live ${live})`;
  } catch (e) {
    verdict = 'FAIL gpu self-test threw: ' + e.message;
  } finally {
    NX = saved.NX; NY = saved.NY; NZ = saved.NZ;
    numSpecies = saved.numSpecies; boundary = saved.boundary;
    backend = saved.backend;
    allocFields(); seed(seedKind); syncSliceUI(); syncSizeUI(); draw();
  }
  console.log(verdict);
  const el = document.getElementById('settingsError');
  if (verdict.startsWith('FAIL')) el.style.color = '#ff8f8f';
  el.textContent += ' · ' + verdict;
}

// Init
allocFields();
seed(seedKind);
syncParamUI();
syncSliceUI();
syncSpeciesUI();
syncSizeUI();
setRenderMode(renderMode);
setTransparentBand(transparentBand);
refreshFills();
refreshSettingsText();
draw();

// WebGL support
function ensureWebGL() {
  if (gl) return;
  // match canvas backing store to device pixel ratio; handle hidden canvas clientWidth=0
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  let w = glCanvas.clientWidth;
  let h = glCanvas.clientHeight;
  if (w < 10 || h < 10) { // fallback when hidden (display:none)
    w = canvas.width; h = canvas.height;
  }
  glCanvas.width = Math.floor(w * dpr);
  glCanvas.height = Math.floor(h * dpr);
  gl = glCanvas.getContext('webgl', { antialias: true });
  if (!gl) {
    console.warn('WebGL not available');
    return;
  }
  initGLResources();
  setupGLInteractions();
}

let glProg, glPosBuf, glColBuf, glProjLoc, glViewLoc, glPosLoc, glColLoc, glPointSizeLoc;

function compileShader(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

function initGLResources() {
  const vs = `
  attribute vec3 aPos;
  attribute vec3 aCol;
  uniform mat4 uProj;
  uniform mat4 uView;
  uniform float uPointSize;
  varying vec3 vCol;
  void main(){
    gl_Position = uProj * uView * vec4(aPos, 1.0);
    vCol = aCol;
    gl_PointSize = uPointSize;
  }`;
  const fs = `
  precision mediump float;
  varying vec3 vCol;
  void main(){
    // circular point sprite
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = dot(uv, uv);
    if (d > 1.0) discard;
    gl_FragColor = vec4(vCol, 1.0);
  }`;
  const vsh = compileShader(gl.VERTEX_SHADER, vs);
  const fsh = compileShader(gl.FRAGMENT_SHADER, fs);
  glProg = gl.createProgram();
  gl.attachShader(glProg, vsh);
  gl.attachShader(glProg, fsh);
  gl.linkProgram(glProg);
  if (!gl.getProgramParameter(glProg, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(glProg));
    return;
  }
  gl.useProgram(glProg);
  glPosLoc = gl.getAttribLocation(glProg, 'aPos');
  glColLoc = gl.getAttribLocation(glProg, 'aCol');
  glProjLoc = gl.getUniformLocation(glProg, 'uProj');
  glViewLoc = gl.getUniformLocation(glProg, 'uView');
  glPointSizeLoc = gl.getUniformLocation(glProg, 'uPointSize');
  glPosBuf = gl.createBuffer();
  glColBuf = gl.createBuffer();
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.039, 0.043, 0.051, 1);   // --void, so the canvas edge disappears
}

function setupGLInteractions() {
  glCanvas.addEventListener('mousedown', (e)=>{ isOrbiting = true; lastMouse.x = e.clientX; lastMouse.y = e.clientY; });
  window.addEventListener('mouseup', ()=>{ isOrbiting = false; });
  window.addEventListener('mousemove', (e)=>{
    if (!isOrbiting) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    lastMouse.x = e.clientX; lastMouse.y = e.clientY;
    camYaw += dx * 0.005;
    camPitch = Math.max(-1.2, Math.min(1.2, camPitch + dy * 0.005));
    draw();
  });
  glCanvas.addEventListener('wheel', (e)=>{
    e.preventDefault();
    camDist = Math.max(20, Math.min(200, camDist + (e.deltaY>0?5:-5)));
    draw();
  }, { passive: false });
}

function makePerspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  // column-major
  return [
    f/aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far+near)*nf, -1,
    0, 0, (2*far*near)*nf, 0
  ];
}

function lookAt(eye, center, up) {
  const zx = eye[0]-center[0], zy = eye[1]-center[1], zz = eye[2]-center[2];
  let zlen = Math.hypot(zx,zy,zz); const zxN=zx/zlen, zyN=zy/zlen, zzN=zz/zlen;
  const xx = up[1]*zzN - up[2]*zyN;
  const xy = up[2]*zxN - up[0]*zzN;
  const xz = up[0]*zyN - up[1]*zxN;
  let xlen = Math.hypot(xx,xy,xz); const xxN=xx/xlen, xyN=xy/xlen, xzN=xz/xlen;
  const yx = zyN*xzN - zzN*xyN;
  const yy = zzN*xxN - zxN*xzN;
  const yz = zxN*xyN - zyN*xxN;
  // column-major view matrix
  return [
    xxN, yx, zxN, 0,
    xyN, yy, zyN, 0,
    xzN, yz, zzN, 0,
    -(xxN*eye[0]+xyN*eye[1]+xzN*eye[2]),
    -(yx*eye[0]+yy*eye[1]+yz*eye[2]),
    -(zxN*eye[0]+zyN*eye[1]+zzN*eye[2]),
    1
  ];
}

function drawWebGL3D() {
  if (!gl) ensureWebGL();
  if (!gl) return;
  glCanvas.style.display = 'block';
  canvas.style.display = 'none';
  gl.viewport(0,0,glCanvas.width, glCanvas.height);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // Build positions and colors from active voxels, centered and scaled
  const positions = [];
  const colors = [];
  const cx = (NX-1)/2, cy = (NY-1)/2, cz = (NZ-1)/2;
  const scale = 1.0; // unit spacing
  for (let z = 0; z < NZ; z++) {
    for (let y = 0; y < NY; y++) {
      for (let x = 0; x < NX; x++) {
        dominantAt(idx(x, y, z));
        if (domE <= 0 || isTransparent(domE)) continue;
        positions.push((x - cx)*scale, (y - cy)*scale, (z - cz)*scale);
        const col = cssToRGB(energyColor(domE, z, domS));
        colors.push(col[0]/255, col[1]/255, col[2]/255);
      }
    }
  }
  const count = positions.length / 3;
  if (count === 0) return;

  gl.bindBuffer(gl.ARRAY_BUFFER, glPosBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(glPosLoc);
  gl.vertexAttribPointer(glPosLoc, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, glColBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(glColLoc);
  gl.vertexAttribPointer(glColLoc, 3, gl.FLOAT, false, 0, 0);

  const aspect = glCanvas.width / glCanvas.height;
  const proj = makePerspective(45*Math.PI/180, aspect, 0.1, 1000.0);
  const eye = [
    camDist*Math.cos(camPitch)*Math.cos(camYaw),
    camDist*Math.sin(camPitch),
    camDist*Math.cos(camPitch)*Math.sin(camYaw)
  ];
  const center = [0,0,0];
  const up = [0,1,0];
  const view = lookAt(eye, center, up);
  gl.uniformMatrix4fv(glProjLoc, false, new Float32Array(proj));
  gl.uniformMatrix4fv(glViewLoc, false, new Float32Array(view));
  const ps = Math.max(2.0, voxelSize * 1.2);
  gl.uniform1f(glPointSizeLoc, ps);

  gl.drawArrays(gl.POINTS, 0, count);
}

function cssToRGB(css) {
  // expects rgb(r,g,b)
  const m = /rgb\((\d+),(\d+),(\d+)\)/.exec(css);
  if (!m) return [255,255,255];
  return [parseInt(m[1],10), parseInt(m[2],10), parseInt(m[3],10)];
}

// Self-check. Open the file with ?selftest=1 to run it; result lands under the
// JSON panel and in the console. Covers the things that fail silently:
// boundary tables, the zero-copy buffer rotation, and species isolation.
function selfTest() {
  const results = [];
  const ok = (name, cond) => {
    results.push((cond ? 'PASS ' : 'FAIL ') + name);
    if (!cond) console.error('FAIL', name);
  };

  const saved = { NX, NY, NZ, numSpecies, boundary, seedKind, sliceAxis,
                  backend, params: { ...params } };
  backend = 'cpu';                       // these checks drive step() directly
  const restore = () => {
    NX = saved.NX; NY = saved.NY; NZ = saved.NZ;
    numSpecies = saved.numSpecies; boundary = saved.boundary;
    seedKind = saved.seedKind; sliceAxis = saved.sliceAxis; params = saved.params;
    backend = saved.backend;
    allocFields(); seed(seedKind); syncSliceUI(); draw();
  };

  try {
    NX = 6; NY = 5; NZ = 4; numSpecies = 1; boundary = 'wrap';
    allocFields();

    // 1. flat index is a bijection over the lattice
    const seen = new Set();
    for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) seen.add(idx(x, y, z));
    ok('idx covers every cell exactly once', seen.size === N && Math.max(...seen) === N - 1);

    // 2. boundary tables resolve the off-lattice slots correctly
    ok('wrap: -1 -> L-1', BX[0] === NX - 1 && BX[NX + 1] === 0);
    boundary = 'reflect'; buildBoundaryTables();
    ok('reflect: -1 -> 0', BX[0] === 0 && BX[NX + 1] === NX - 1);
    boundary = 'absorb'; buildBoundaryTables();
    ok('absorb: outside marked -1', BX[0] === -1 && BX[NX + 1] === -1);

    // 3. a full lattice: a corner cell sees 26 neighbours wrapped, 7 absorbed
    const cornerActive = () => { computeRowBases(0, 0); gather(0, 0, BX[0], BX[2]); return nbActive; };
    boundary = 'wrap'; buildBoundaryTables(); clearAll();
    grid.fill(200);
    const wrapActive = cornerActive();
    boundary = 'absorb'; buildBoundaryTables();
    ok('absorb corner loses neighbours', wrapActive === 26 && cornerActive() === 7);

    // 4. buffer rotation: echo is t-2, prev is t-1, no buffer aliases another
    boundary = 'wrap'; buildBoundaryTables(); clearAll();
    for (let i = 0; i < N; i++) grid[i] = (i % 7) * 20;
    const t0 = Float32Array.from(grid);
    step();
    const t1 = Float32Array.from(grid);
    step();
    ok('echo holds t-2', echo.every((v, i) => v === t0[i]));
    ok('prev holds t-1', prev.every((v, i) => v === t1[i]));
    ok('four distinct buffers', new Set([grid.buffer, prev.buffer, echo.buffer, next.buffer]).size === 4);

    // 5. species planes do not bleed into each other
    numSpecies = 2; allocFields(); clearAll();
    params.inhibit = 0;
    for (let i = 0; i < N; i++) grid[N + i] = 180;   // species 2 only
    step(); step();
    let leaked = 0;
    for (let i = 0; i < N; i++) if (grid[i] !== 0) leaked++;
    ok('empty species stays empty', leaked === 0);

    // 6. inhibition suppresses a species sharing space with a rival
    const runBoth = (inh) => {
      clearAll();
      for (let i = 0; i < N; i++) { grid[i] = 150; grid[N + i] = 150; }
      params.inhibit = inh;
      step();
      return grid[0];
    };
    ok('inhibition lowers energy under a rival', runBoth(1.0) < runBoth(0));

    // 7. every seed kind puts something on the board
    numSpecies = 2; allocFields();
    for (const k of ['random', 'sphere', 'plane', 'filament', 'symmetric']) {
      seed(k);
      let n = 0;
      for (let i = 0; i < grid.length; i++) if (grid[i] > 0) n++;
      ok(`seed "${k}" produces live cells`, n > 0);
    }

    // 8. slice addressing stays in bounds on all three axes
    let inBounds = true;
    for (const ax of ['xy', 'xz', 'yz']) {
      sliceAxis = ax;
      const [uN, vN] = sliceUV();
      for (let k = 0; k < sliceDepth(); k++)
        for (let v = 0; v < vN; v++)
          for (let u = 0; u < uN; u++) {
            const ci = sliceIdx(u, v, k);
            if (!(ci >= 0 && ci < N)) inBounds = false;
          }
    }
    ok('slice addressing in bounds on all axes', inBounds);
  } finally {
    restore();
  }

  const failed = results.filter(r => r.startsWith('FAIL'));
  console.log(results.join('\n'));
  const el = document.getElementById('settingsError');
  el.style.color = failed.length ? '#ff8f8f' : '#7ef0b0';
  el.textContent = `selftest: ${results.length - failed.length}/${results.length} passed` +
    (failed.length ? ' -- ' + failed.join('; ') : '');
}

if (location.search.includes('selftest')) selfTest();

// Probe for a device last: everything above already works without one.
initGPU().then(ok => {
  const btn = document.querySelector('#backendSeg button[data-backend="gpu"]');
  if (!ok) {
    btn.title = navigator.gpu ? 'No WebGPU adapter on this machine'
                              : 'This browser has no WebGPU';
    return;
  }
  btn.disabled = false;
  btn.title = 'Run the rule as a compute shader';
  setBackend('gpu');
  draw();
  if (location.search.includes('selftest')) gpuSelfTest();
});
