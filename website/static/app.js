/* Second Life AI — single-page front end over the real classifier API. */

const CLASS_LABEL = {
  cardboard: "Cardboard", glass: "Glass", metal: "Metal",
  paper: "Paper", plastic: "Plastic", trash: "General waste",
};
const BIN_COLOR = {
  paper: "#B98524", glass: "#3E7C8C", metal: "#6E7B86",
  plastic: "#C0803A", trash: "#7A7A73",
};
const SAMPLE_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];

const state = { current: null };          // last identification, shared result -> simulator
const app = document.getElementById("app");

/* ---------------- record (localStorage) ---------------- */
const RECORD_KEY = "sla_record_v1";
const getRecord = () => JSON.parse(localStorage.getItem(RECORD_KEY) || "[]");
function addRecord(cls) {
  const r = getRecord();
  r.unshift({ cls, ts: Date.now() });
  localStorage.setItem(RECORD_KEY, JSON.stringify(r.slice(0, 100)));
}

/* ---------------- router ---------------- */
const routes = {
  "": initHome, "/": initHome,
  "/identify": initIdentify,
  "/result": initResult,
  "/simulator": initSimulator,
  "/explore": initExplore,
  "/about": initAbout,
};

function render(name) {
  const tpl = document.getElementById("view-" + name);
  app.replaceChildren(tpl.content.cloneNode(true));
}

function route() {
  const path = location.hash.replace(/^#/, "") || "/";
  const key = path === "/" ? "/" : path;
  // guard: result/simulator need an identification first
  if ((key === "/result" || key === "/simulator") && !state.current) {
    location.hash = "#/identify";
    return;
  }
  (routes[key] || initHome)();
  document.querySelectorAll(".nav-links a").forEach((a) => {
    a.classList.toggle("active", a.getAttribute("href") === "#" + key);
  });
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", route);
document.addEventListener("click", (e) => {
  const link = e.target.closest("a[data-link]");
  if (link) { /* let the hash change naturally */ }
});

/* ---------------- views ---------------- */
function initHome() { render("home"); }

function initIdentify() {
  render("identify");
  const fileInput = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });
  ["dragover", "dragenter"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, () => dropzone.classList.remove("drag")));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });

  const row = document.getElementById("samples-row");
  SAMPLE_CLASSES.forEach((c) => {
    const b = document.createElement("button");
    b.title = "Try the " + CLASS_LABEL[c].toLowerCase() + " sample";
    b.innerHTML = `<img src="img/sample_${c}.jpg" alt="${CLASS_LABEL[c]} sample" />`;
    b.addEventListener("click", () => identifyFromUrl(`img/sample_${c}.jpg`));
    row.appendChild(b);
  });
}

function handleFile(file) {
  const url = URL.createObjectURL(file);
  const fd = new FormData();
  fd.append("image", file);
  identify(fd, url);
}

async function identifyFromUrl(url) {
  const blob = await (await fetch(url)).blob();
  const fd = new FormData();
  fd.append("image", blob, "sample.jpg");
  identify(fd, url);
}

function setStatus(msg, isError = false) {
  const el = document.getElementById("identify-status");
  if (!el) return;
  el.hidden = false;
  el.className = "status" + (isError ? " error" : "");
  el.innerHTML = isError ? msg : `<span class="spinner"></span>${msg}`;
}

async function identify(formData, displayUrl) {
  setStatus("Looking closely…");
  try {
    const res = await fetch("/api/identify", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    state.current = { ...data, imageUrl: displayUrl };
    addRecord(data.top);
    location.hash = "#/result";
  } catch (err) {
    setStatus("Couldn't identify that — " + err.message, true);
  }
}

function initResult() {
  render("result");
  const { top, confidence, predictions, content, imageUrl } = state.current;
  document.getElementById("r-image").src = imageUrl;
  document.getElementById("r-name").textContent = content.name;
  document.getElementById("r-tagline").textContent = content.tagline;

  const pct = Math.round(confidence * 100);
  const alts = predictions.slice(1, 3)
    .map((p) => `${CLASS_LABEL[p.class]} ${Math.round(p.prob * 100)}%`).join("   ·   ");
  document.getElementById("r-confidence").innerHTML = `
    <div class="conf-row"><span>How sure I am</span><span class="conf-pct">${pct}%</span></div>
    <div class="conf-bar"><div class="conf-fill" style="width:0%"></div></div>
    <div class="conf-alts">also considered: ${alts}</div>`;
  requestAnimationFrame(() =>
    (document.querySelector(".conf-fill").style.width = pct + "%"));

  const binBlock = document.getElementById("r-bin-block");
  binBlock.style.setProperty("--bin-color", BIN_COLOR[content.bin_key] || "#35674A");
  document.getElementById("r-bin").textContent = content.bin;
  document.getElementById("r-prep").textContent = content.prep[0];

  document.getElementById("r-material").textContent = content.material;
  document.getElementById("r-recyclable").textContent = content.recyclable.split(" — ")[0];
  document.getElementById("r-decompose").textContent = content.decompose_landfill;

  fillChips("r-secondlife", content.second_life);
  fillChips("r-related", content.related);
  document.getElementById("r-didyouknow").textContent = content.did_you_know;
}

function fillChips(id, items) {
  const ul = document.getElementById(id);
  ul.replaceChildren(...items.map((t) => { const li = document.createElement("li"); li.textContent = t; return li; }));
}

function initSimulator() {
  render("simulator");
  const { content } = state.current;
  document.getElementById("s-name").textContent = content.name.toLowerCase();

  const fork = document.getElementById("s-fork");
  const outcome = document.getElementById("s-outcome");
  const card = document.getElementById("s-outcome-card");
  const tag = document.getElementById("s-outcome-tag");
  const body = document.getElementById("s-outcome-body");
  const rewriteWrap = document.getElementById("s-rewrite");
  const rewriteBtn = document.getElementById("s-rewrite-btn");
  const rewritten = document.getElementById("s-rewritten");

  function reveal(good) {
    card.classList.toggle("bad", !good);
    tag.textContent = good ? "One month later" : "Years from now";
    body.textContent = good ? content.future_good : content.future_bad;
    outcome.hidden = false;
    rewritten.hidden = true;
    if (good) {
      rewriteWrap.hidden = true;
      showRewritten();
    } else {
      rewriteWrap.hidden = false;
    }
  }
  function showRewritten() {
    document.getElementById("s-rewritten-body").textContent = content.rewrite;
    rewritten.hidden = false;
  }

  fork.querySelectorAll(".path").forEach((btn) => {
    btn.addEventListener("click", () => {
      const good = btn.dataset.path === "good";
      fork.querySelectorAll(".path").forEach((p) => {
        p.classList.toggle("chosen", p === btn);
        p.classList.toggle("dim", p !== btn);
      });
      reveal(good);
    });
  });
  rewriteBtn.addEventListener("click", () => {
    rewriteWrap.hidden = true;
    document.querySelector(".path-good").classList.remove("dim");
    document.querySelector(".path-good").classList.add("chosen");
    showRewritten();
  });
}

async function initExplore() {
  render("explore");
  try {
    const { spotlight } = await (await fetch("/api/spotlight")).json();
    document.getElementById("sp-title").textContent = spotlight.title;
    document.getElementById("sp-body").textContent = spotlight.body;
  } catch { document.getElementById("sp-title").textContent = "Come back tomorrow"; }

  const record = getRecord();
  const stats = document.getElementById("record-stats");
  const list = document.getElementById("record-list");
  const empty = document.getElementById("record-empty");
  if (!record.length) { empty.hidden = false; return; }

  const unique = new Set(record.map((r) => r.cls)).size;
  stats.innerHTML = `
    <div class="stat"><div class="stat-num">${record.length}</div><div class="stat-label">Items identified</div></div>
    <div class="stat"><div class="stat-num">${unique}</div><div class="stat-label">Materials met</div></div>
    <div class="stat"><div class="stat-num">${Math.round(unique / 6 * 100)}%</div><div class="stat-label">Of all six</div></div>`;
  list.replaceChildren(...record.slice(0, 20).map((r) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="record-cls">${CLASS_LABEL[r.cls]}</span>
      <span class="record-when">${new Date(r.ts).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</span>`;
    return li;
  }));
}

async function initAbout() {
  render("about");
  try {
    const m = await (await fetch("/api/model")).json();
    const kind = m.is_ensemble ? `${m.members}-model ensemble` : "single model";
    document.getElementById("about-live").textContent =
      `Serving now: ${m.model} · ${kind} · ${m.img_size}px · on ${m.device}.`;
  } catch { /* leave blank */ }
}

/* ---------------- boot ---------------- */
async function boot() {
  try {
    const m = await (await fetch("/api/model")).json();
    const kind = m.is_ensemble ? `${m.members}-model ensemble` : m.model;
    document.getElementById("foot-model").textContent = `${kind} · ${m.val_accuracy_tta} val`;
  } catch { document.getElementById("foot-model").textContent = "model offline"; }
  route();
}
boot();
