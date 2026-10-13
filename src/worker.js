/**
 * worker.js — runs training on a background thread so the page stays responsive.
 *
 * Messages in:  {type: "init", gen, cfg} | {type: "pause"} | {type: "resume"}
 * Messages out: {type: "snapshot", gen, snap} | {type: "error", gen, message}
 *
 * `gen` is a generation counter chosen by the page. Snapshots from an older run are
 * ignored on the page side, so a restart never shows stale results.
 */
importScripts("engine.js", "eigensolver.js", "solver.js", "potentials.js", "trainer.js");

let trainer = null, gen = 0, running = false, timer = 0;

function tick() {
  clearTimeout(timer);
  if (!trainer) return;
  try {
    if (running && !trainer.done) trainer.run(45);
    postMessage({ type: "snapshot", gen, snap: trainer.snapshot() });
  } catch (err) {
    running = false;
    postMessage({ type: "error", gen, message: err.message });
    return;
  }
  if (running && !trainer.done) timer = setTimeout(tick, 0);
}

onmessage = e => {
  const m = e.data;
  if (m.type === "init") {
    gen = m.gen;
    try {
      trainer = new NQ.Trainer(m.cfg);
    } catch (err) {
      postMessage({ type: "error", gen, message: err.message });
      return;
    }
    running = true;
    tick();
  } else if (m.type === "pause") {
    running = false;
    clearTimeout(timer);
  } else if (m.type === "resume" && trainer) {
    if (trainer.done) trainer.extend();
    running = true;
    tick();
  }
};
