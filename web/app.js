import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const css = getComputedStyle(document.documentElement);
const token = (name) => css.getPropertyValue(name).trim();
const COLORS = {
  accent: token("--accent"),
  covered: token("--covered"),
  uncovered: token("--uncovered"),
  ghost: token("--ghost"),
  ap: token("--ap"),
  bs: token("--bs"),
  surface: token("--surface-1"),
  textPrimary: token("--text-primary"),
  textMuted: token("--text-muted"),
};
// Fixed policy -> series slot; colour follows the policy, never its rank.
const SERIES = {
  MADDPG: token("--series-1"),
  Greedy: token("--series-2"),
  Static: token("--series-3"),
  Random: token("--series-4"),
  "Local greedy": token("--series-5"),
};
const POLICY_HINTS = {
  MADDPG:
    "Trained multi-agent RL. Each drone sees only its 6 nearest users and its teammates; one shared actor, centralised critic.",
  Greedy: "Centralised baseline: sees every user and runs k-means; each drone flies to its cluster centre at the best altitude.",
  "Local greedy":
    "Decentralised baseline with exactly the information one MADDPG drone has: fly toward nearby users no teammate covers.",
  Static: "Baseline: drones hover wherever they start.",
  Random: "Baseline: every drone makes a random move each step.",
};
const MODE_LABELS = { predicted: "Predicted from RF", oracle: "True positions" };
const STEPS_PER_SECOND = 2;
const TRAIL_LENGTH = 40;

const $ = (id) => document.getElementById(id);
const state = { policy: null, mode: null, episode: 0, t: 0, playing: true, speed: 1, lastStep: -1 };
let data = null;
let run = null;

/* ---------------- data ---------------- */

async function loadData() {
  try {
    const response = await fetch("data/replays.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } catch (error) {
    const status = $("status");
    status.innerHTML = `<div><p>Couldn't load <code>data/replays.json</code> (${error.message}).</p>
      <p>Generate it with <code>python scripts/export_replays.py</code>, then serve this folder with
      <code>python -m http.server -d web</code>. Opening the HTML file directly won't work.</p></div>`;
    status.classList.add("show");
    throw error;
  }
}

function findRun() {
  return data.runs.find(
    (r) => r.policy === state.policy && r.positions === state.mode && r.episode === state.episode,
  );
}

const frameCount = () => run.frames.uavs.length;

/* ---------------- three.js scene ---------------- */

const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0b0f14");
scene.fog = new THREE.Fog("#0b0f14", 900, 2200);
const camera = new THREE.PerspectiveCamera(45, 1, 1, 5000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 150;
controls.maxDistance = 1600;

scene.add(new THREE.HemisphereLight("#b9d4ff", "#1a1f26", 1.1));
const sun = new THREE.DirectionalLight("#ffffff", 1.4);
sun.position.set(300, 600, 200);
scene.add(sun);

let half = 250;
const toWorld = (x, y, h = 0) => new THREE.Vector3(x - half, h, half - y);

const layers = {
  cones: new THREE.Group(),
  ghosts: new THREE.Group(),
  trails: new THREE.Group(),
  infra: new THREE.Group(),
};
Object.values(layers).forEach((g) => scene.add(g));

let users, ghosts, ghostLines, drones = [], cones = [], trails = [];

function buildGround(area) {
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(area, area),
    new THREE.MeshStandardMaterial({ color: "#151b22", roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  const outside = new THREE.Mesh(
    new THREE.PlaneGeometry(area * 8, area * 8),
    new THREE.MeshBasicMaterial({ color: "#0d1117" }),
  );
  outside.rotation.x = -Math.PI / 2;
  outside.position.y = -0.5;
  scene.add(outside);

  const grid = new THREE.GridHelper(area, 10, "#2a323c", "#1f262e");
  grid.position.y = 0.05;
  scene.add(grid);

  const border = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints([
      toWorld(0, 0, 0.2), toWorld(area, 0, 0.2), toWorld(area, area, 0.2), toWorld(0, area, 0.2),
    ]),
    new THREE.LineBasicMaterial({ color: "#3b4552" }),
  );
  scene.add(border);
}

function buildInfrastructure(meta) {
  const poleMat = new THREE.MeshStandardMaterial({ color: "#4a5260", roughness: 0.7 });
  for (const [x, y, h] of meta.access_points) {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, h * 2), poleMat);
    pole.position.copy(toWorld(x, y, h));
    const head = new THREE.Mesh(
      new THREE.SphereGeometry(3, 16, 12),
      new THREE.MeshStandardMaterial({ color: COLORS.ap, emissive: COLORS.ap, emissiveIntensity: 0.6 }),
    );
    head.position.copy(toWorld(x, y, h * 2 + 2));
    layers.infra.add(pole, head);
  }
  for (const [x, y, h] of meta.base_stations) {
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(1.2, 2.4, h), poleMat);
    mast.position.copy(toWorld(x, y, h / 2));
    const top = new THREE.Mesh(
      new THREE.ConeGeometry(5, 9, 3),
      new THREE.MeshStandardMaterial({ color: COLORS.bs, emissive: COLORS.bs, emissiveIntensity: 0.5 }),
    );
    top.position.copy(toWorld(x, y, h + 4.5));
    layers.infra.add(mast, top);
  }
}

function labelSprite(text) {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgba(11,15,20,0.85)";
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = COLORS.accent;
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  ctx.font = "600 30px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, size / 2, size / 2 + 1);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }),
  );
  sprite.scale.set(14, 14, 1);
  sprite.position.y = 16;
  return sprite;
}

function buildDrone(index) {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: "#d8e3ee", metalness: 0.3, roughness: 0.4 });
  const accentMat = new THREE.MeshStandardMaterial({
    color: COLORS.accent, emissive: COLORS.accent, emissiveIntensity: 0.8,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(10, 3, 10), bodyMat);
  group.add(body);
  const light = new THREE.Mesh(new THREE.SphereGeometry(2, 12, 8), accentMat);
  light.position.y = -2;
  group.add(light);
  const rotors = [];
  for (let k = 0; k < 4; k++) {
    const angle = Math.PI / 4 + (k * Math.PI) / 2;
    const arm = new THREE.Mesh(new THREE.BoxGeometry(12, 1, 1.2), bodyMat);
    arm.rotation.y = -angle;
    arm.position.set(Math.cos(angle) * 6, 0, Math.sin(angle) * 6);
    group.add(arm);
    const rotor = new THREE.Mesh(
      new THREE.CylinderGeometry(5, 5, 0.4, 20),
      new THREE.MeshStandardMaterial({ color: "#9fb3c8", transparent: true, opacity: 0.55 }),
    );
    rotor.position.set(Math.cos(angle) * 11, 1.6, Math.sin(angle) * 11);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(9, 0.5, 1), accentMat);
    rotor.add(blade);
    group.add(rotor);
    rotors.push(rotor);
  }
  group.add(labelSprite(String(index + 1)));
  group.userData.rotors = rotors;
  scene.add(group);
  return group;
}

function buildCone() {
  const group = new THREE.Group();
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(1, 1, 64, 1, true),
    new THREE.MeshBasicMaterial({
      color: COLORS.accent, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false,
    }),
  );
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(1, 64),
    new THREE.MeshBasicMaterial({ color: COLORS.accent, transparent: true, opacity: 0.1, depthWrite: false }),
  );
  disc.rotation.x = -Math.PI / 2;
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.985, 1, 96),
    new THREE.MeshBasicMaterial({ color: COLORS.accent, transparent: true, opacity: 0.6, depthWrite: false }),
  );
  ring.rotation.x = -Math.PI / 2;
  group.add(cone, disc, ring);
  group.userData = { cone, disc, ring };
  layers.cones.add(group);
  return group;
}

function buildTrail() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL_LENGTH * 3 + 3), 3));
  const line = new THREE.Line(
    geometry,
    new THREE.LineBasicMaterial({ color: COLORS.accent, transparent: true, opacity: 0.45 }),
  );
  line.frustumCulled = false;
  layers.trails.add(line);
  return line;
}

function buildScene(meta) {
  half = meta.area_size / 2;
  buildGround(meta.area_size);
  buildInfrastructure(meta);

  users = new THREE.InstancedMesh(
    new THREE.SphereGeometry(4.5, 16, 12),
    new THREE.MeshStandardMaterial({ roughness: 0.5 }),
    meta.n_users,
  );
  scene.add(users);

  ghosts = new THREE.InstancedMesh(
    new THREE.RingGeometry(4.5, 6.5, 24),
    new THREE.MeshBasicMaterial({ color: COLORS.ghost, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }),
    meta.n_users,
  );
  layers.ghosts.add(ghosts);
  ghostLines = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute(
      "position", new THREE.BufferAttribute(new Float32Array(meta.n_users * 6), 3),
    ),
    new THREE.LineBasicMaterial({ color: COLORS.ghost, transparent: true, opacity: 0.35 }),
  );
  ghostLines.frustumCulled = false;
  layers.ghosts.add(ghostLines);

  for (let i = 0; i < meta.n_uavs; i++) {
    drones.push(buildDrone(i));
    cones.push(buildCone());
    trails.push(buildTrail());
  }

  const dist = meta.area_size * 1.45;
  camera.position.set(-dist * 0.2, dist * 0.8, dist * 0.85);
  controls.target.set(0, 0, 20);
}

/* ---------------- per-frame scene update ---------------- */

const tmpMatrix = new THREE.Matrix4();
const tmpQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
const tmpScale = new THREE.Vector3(1, 1, 1);
const identity = new THREE.Quaternion();
const tmpColor = new THREE.Color();
const lerp = (a, b, f) => a + (b - a) * f;

function sample(key, t) {
  const frames = run.frames[key];
  const i = Math.min(Math.floor(t), frames.length - 1);
  const j = Math.min(i + 1, frames.length - 1);
  return { a: frames[i], b: frames[j], f: t - i, i };
}

function updateScene(t) {
  const { a: ua, b: ub, f } = sample("uavs", t);
  const { a: ra, b: rb } = sample("radius", t);
  const step = Math.min(Math.floor(t), frameCount() - 1);

  ua.forEach((pa, k) => {
    const pb = ub[k];
    const x = lerp(pa[0], pb[0], f);
    const y = lerp(pa[1], pb[1], f);
    const h = lerp(pa[2], pb[2], f);
    drones[k].position.copy(toWorld(x, y, h));
    const r = lerp(ra[k], rb[k], f);
    const { cone, disc, ring } = cones[k].userData;
    cones[k].position.copy(toWorld(x, y, 0));
    cone.scale.set(r, h, r);
    cone.position.y = h / 2;
    disc.scale.setScalar(r);
    disc.position.y = 0.3;
    ring.scale.setScalar(r);
    ring.position.y = 0.4;

    const attr = trails[k].geometry.attributes.position;
    const start = Math.max(0, step - TRAIL_LENGTH + 1);
    let n = 0;
    for (let s = start; s <= step; s++, n++) {
      const p = run.frames.uavs[s][k];
      const w = toWorld(p[0], p[1], p[2]);
      attr.setXYZ(n, w.x, w.y, w.z);
    }
    const now = drones[k].position;
    attr.setXYZ(n++, now.x, now.y, now.z);
    attr.needsUpdate = true;
    trails[k].geometry.setDrawRange(0, n);
  });

  const { a: pa, b: pb } = sample("users", t);
  const { a: oa, b: ob } = sample("observed", t);
  const covered = run.frames.covered[step];
  const linePos = ghostLines.geometry.attributes.position;
  for (let u = 0; u < pa.length; u++) {
    const user = toWorld(lerp(pa[u][0], pb[u][0], f), lerp(pa[u][1], pb[u][1], f), 4.5);
    tmpMatrix.compose(user, identity, tmpScale);
    users.setMatrixAt(u, tmpMatrix);
    users.setColorAt(u, tmpColor.set(covered[u] ? COLORS.covered : COLORS.uncovered));

    const ghost = toWorld(lerp(oa[u][0], ob[u][0], f), lerp(oa[u][1], ob[u][1], f), 0.6);
    tmpMatrix.compose(ghost, tmpQuat, tmpScale);
    ghosts.setMatrixAt(u, tmpMatrix);
    linePos.setXYZ(u * 2, user.x, 0.6, user.z);
    linePos.setXYZ(u * 2 + 1, ghost.x, ghost.y, ghost.z);
  }
  users.instanceMatrix.needsUpdate = true;
  users.instanceColor.needsUpdate = true;
  ghosts.instanceMatrix.needsUpdate = true;
  linePos.needsUpdate = true;
}

/* ---------------- panel ---------------- */

function segmented(container, options, current, onPick) {
  container.innerHTML = "";
  for (const { value, label } of options) {
    const button = document.createElement("button");
    button.textContent = label;
    button.dataset.value = value;
    button.classList.toggle("active", value === current);
    button.setAttribute("aria-pressed", value === current);
    button.addEventListener("click", () => onPick(value));
    container.appendChild(button);
  }
}

function renderControls() {
  segmented($("policy"), data.meta.policies.map((p) => ({ value: p, label: p })), state.policy, (p) => {
    state.policy = p;
    selectRun(false);
  });
  segmented($("mode"), data.meta.modes.map((m) => ({ value: m, label: MODE_LABELS[m] })), state.mode, (m) => {
    state.mode = m;
    selectRun(false);
  });
  const episodes = Array.from({ length: data.meta.episodes }, (_, i) => ({ value: i, label: `#${i + 1}` }));
  segmented($("episode"), episodes, state.episode, (e) => {
    state.episode = e;
    selectRun(true);
  });
  $("policy-hint").textContent = POLICY_HINTS[state.policy] ?? "";
  layers.ghosts.visible = $("layer-ghosts").checked && state.mode === "predicted";
}

function renderResults() {
  const summary = data.summary[state.mode];
  const head = "<tr><th>Policy</th><th>Coverage</th><th>Flown per step</th></tr>";
  const rows = data.meta.policies
    .map((p) => {
      const { coverage, distance_m: distance } = summary[p];
      return `<tr class="${p === state.policy ? "current" : ""}"><td>${p}</td>
        <td>${(coverage * 100).toFixed(1)}%</td><td>${distance.toFixed(1)} m</td></tr>`;
    })
    .join("");
  $("results").innerHTML = head + rows;
  $("results-sub").textContent =
    `${MODE_LABELS[state.mode]} · mean over ${data.meta.episodes} held-out episodes`;
  const m = data.meta;
  $("predictor-line").textContent =
    m.rmse_meters != null
      ? `Position predictor test error: ${m.rmse_meters} m RMSE (guessing the area centre: ${m.center_guess_rmse_meters} m). ` +
        "Flown per step is each drone's average movement, a proxy for battery use."
      : "";
}

function renderDroneList(step) {
  const uavs = run.frames.uavs[step];
  const battery = run.frames.battery[step];
  $("drone-list").innerHTML = uavs
    .map((p, k) => {
      const pct = Math.round(battery[k] * 100);
      return `<li><span class="name">UAV ${k + 1}</span><span class="alt">${Math.round(p[2])} m</span>
        <span class="battery ${pct < 20 ? "low" : ""}" role="meter" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="UAV ${k + 1} battery"><i style="width:${pct}%"></i></span>
        <span class="pct">${pct}%</span></li>`;
    })
    .join("");
}

function renderHud(step) {
  const coverage = run.frames.coverage[step];
  const covered = run.frames.covered[step].reduce((s, v) => s + v, 0);
  $("coverage-pct").textContent = (coverage * 100).toFixed(0);
  $("coverage-count").textContent = `${covered} of ${data.meta.n_users} users`;
  $("step-label").textContent = `${step} / ${frameCount() - 1}`;
  $("error-label").textContent =
    state.mode === "predicted" ? `${run.frames.error_m[step].toFixed(1)} m` : "0 m · true positions";
}

/* ---------------- chart ---------------- */

const chart = $("chart");
const chartCtx = chart.getContext("2d");
const PAD = { left: 34, right: 12, top: 10, bottom: 22 };
let hoverStep = null;

function episodeRuns() {
  return data.meta.policies
    .map((p) => data.runs.find((r) => r.policy === p && r.positions === state.mode && r.episode === state.episode))
    .filter(Boolean);
}

function chartGeometry() {
  const w = chart.clientWidth;
  const h = chart.clientHeight;
  const steps = frameCount() - 1;
  const x = (s) => PAD.left + (s / steps) * (w - PAD.left - PAD.right);
  const y = (v) => PAD.top + (1 - v) * (h - PAD.top - PAD.bottom);
  return { w, h, steps, x, y };
}

function drawChart() {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const { w, h, steps, x, y } = chartGeometry();
  if (chart.width !== Math.round(w * dpr)) {
    chart.width = Math.round(w * dpr);
    chart.height = Math.round(h * dpr);
  }
  const ctx = chartCtx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.textBaseline = "middle";
  for (const v of [0, 0.5, 1]) {
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y(v));
    ctx.lineTo(w - PAD.right, y(v));
    ctx.stroke();
    ctx.fillStyle = COLORS.textMuted;
    ctx.textAlign = "right";
    ctx.fillText(`${v * 100}%`, PAD.left - 6, y(v));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("step", (PAD.left + w - PAD.right) / 2, h - 4);

  const all = episodeRuns();
  const ordered = [...all.filter((r) => r.policy !== state.policy), ...all.filter((r) => r.policy === state.policy)];
  for (const r of ordered) {
    const active = r.policy === state.policy;
    ctx.strokeStyle = SERIES[r.policy];
    ctx.globalAlpha = active ? 1 : 0.4;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    r.frames.coverage.forEach((v, s) => (s === 0 ? ctx.moveTo(x(s), y(v)) : ctx.lineTo(x(s), y(v))));
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // Playhead.
  const px = x(Math.min(state.t, steps));
  ctx.strokeStyle = COLORS.textPrimary;
  ctx.globalAlpha = 0.6;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px, PAD.top);
  ctx.lineTo(px, h - PAD.bottom);
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Hover crosshair + markers.
  if (hoverStep != null) {
    const hx = x(hoverStep);
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(hx, PAD.top);
    ctx.lineTo(hx, h - PAD.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    for (const r of all) {
      ctx.fillStyle = SERIES[r.policy];
      ctx.strokeStyle = COLORS.surface;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(hx, y(r.frames.coverage[hoverStep]), 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function renderChartLegend() {
  const legend = $("chart-legend");
  legend.innerHTML = "";
  for (const r of episodeRuns()) {
    const mean = r.frames.coverage.slice(1).reduce((s, v) => s + v, 0) / (r.frames.coverage.length - 1);
    const button = document.createElement("button");
    button.className = r.policy === state.policy ? "active" : "";
    button.innerHTML = `<i class="line-key" style="background:${SERIES[r.policy]}"></i>${r.policy} <span>${(mean * 100).toFixed(0)}%</span>`;
    button.title = `Show ${r.policy}`;
    button.addEventListener("click", () => {
      state.policy = r.policy;
      selectRun(false);
    });
    legend.appendChild(button);
  }
}

function stepFromEvent(event) {
  const rect = chart.getBoundingClientRect();
  const { w, steps } = chartGeometry();
  const frac = (event.clientX - rect.left - PAD.left) / (w - PAD.left - PAD.right);
  return Math.max(0, Math.min(steps, Math.round(frac * steps)));
}

chart.addEventListener("pointermove", (event) => {
  if (!run) return;
  hoverStep = stepFromEvent(event);
  const tip = $("chart-tip");
  const rows = episodeRuns()
    .map((r) => ({ p: r.policy, v: r.frames.coverage[hoverStep] }))
    .sort((a, b) => b.v - a.v)
    .map(({ p, v }) => `<div class="tip-row"><i class="line-key" style="background:${SERIES[p]}"></i>${p}<b>${(v * 100).toFixed(0)}%</b></div>`)
    .join("");
  tip.innerHTML = `<div class="tip-head">Step ${hoverStep} · click to jump</div>${rows}`;
  tip.hidden = false;
  const { x, w } = chartGeometry();
  const left = x(hoverStep) + 12;
  tip.style.left = `${left + 140 > w ? x(hoverStep) - 152 : left}px`;
  drawChart();
});
chart.addEventListener("pointerleave", () => {
  hoverStep = null;
  $("chart-tip").hidden = true;
  drawChart();
});
chart.addEventListener("click", (event) => seek(stepFromEvent(event)));

/* ---------------- playback ---------------- */

function selectRun(resetTime) {
  run = findRun();
  if (!run) return;
  if (resetTime) state.t = 0;
  state.t = Math.min(state.t, frameCount() - 1);
  $("scrubber").max = frameCount() - 1;
  state.lastStep = -1;
  renderControls();
  renderResults();
  renderChartLegend();
  refresh();
}

function refresh() {
  const step = Math.min(Math.floor(state.t), frameCount() - 1);
  updateScene(state.t);
  if (step !== state.lastStep) {
    renderHud(step);
    renderDroneList(step);
    state.lastStep = step;
  }
  $("scrubber").value = state.t;
  drawChart();
}

function seek(t) {
  state.t = t;
  refresh();
}

function setPlaying(playing) {
  state.playing = playing;
  $("icon-play").hidden = playing;
  $("icon-pause").hidden = !playing;
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
}

$("play").addEventListener("click", () => {
  if (!state.playing && state.t >= frameCount() - 1) state.t = 0;
  setPlaying(!state.playing);
});
$("scrubber").addEventListener("input", (e) => {
  setPlaying(false);
  seek(parseFloat(e.target.value));
});
$("speed").addEventListener("click", (e) => {
  const button = e.target.closest("button");
  if (!button) return;
  state.speed = parseFloat(button.dataset.speed);
  $("speed").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === button));
});
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, button")) return;
  if (e.code === "Space") {
    e.preventDefault();
    $("play").click();
  }
});

for (const [id, layer] of [["layer-cones", "cones"], ["layer-trails", "trails"], ["layer-infra", "infra"]]) {
  $(id).addEventListener("change", (e) => (layers[layer].visible = e.target.checked));
}
$("layer-ghosts").addEventListener("change", (e) => {
  layers.ghosts.visible = e.target.checked && state.mode === "predicted";
});

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas.parentElement;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // Portrait screens: widen the field of view so the whole area stays in frame.
  camera.fov = camera.aspect < 1 ? 45 / Math.max(camera.aspect, 0.5) : 45;
  camera.updateProjectionMatrix();
  if (run) drawChart();
}
new ResizeObserver(resize).observe(canvas.parentElement);

const clock = new THREE.Clock();
function animate() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (run && state.playing) {
    state.t += dt * STEPS_PER_SECOND * state.speed;
    const last = frameCount() - 1;
    if (state.t >= last) {
      // Loop the episode after a short pause at the end.
      state.t = state.t >= last + 1.5 ? 0 : state.t;
    }
    refresh();
    for (const drone of drones) {
      for (const rotor of drone.userData.rotors) rotor.rotation.y += dt * 40;
    }
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

/* ---------------- boot ---------------- */

data = await loadData();
buildScene(data.meta);
// Shareable links: ?policy=Greedy&mode=oracle&episode=2&t=40
const params = new URLSearchParams(location.search);
state.policy = data.meta.policies.includes(params.get("policy")) ? params.get("policy") : data.meta.policies[0];
state.mode = data.meta.modes.includes(params.get("mode")) ? params.get("mode") : data.meta.modes[0];
const episodeParam = parseInt(params.get("episode"), 10);
state.episode = episodeParam >= 1 && episodeParam <= data.meta.episodes ? episodeParam - 1 : 0;
resize();
selectRun(true);
if (params.has("t")) {
  seek(Math.max(0, Math.min(parseFloat(params.get("t")) || 0, frameCount() - 1)));
  setPlaying(false);
}
animate();
