/* Failed Star — diagrams.js
   The sampling explorer: real softmax over a fixed toy logit vector, reshaped live
   by temperature / top-k / top-p. No framework, no build step. (Loaded only on the
   diagrams page; everything is guarded by #sampler-ish elements existing.) */

(function () {
  "use strict";

  var dist = document.getElementById("dist");
  if (!dist) return;

  // Toy next-token candidates for "The cat sat on the ___" with raw logits.
  var TOKENS = [
    { t: "mat",      l: 3.1 },
    { t: "floor",    l: 2.4 },
    { t: "rug",      l: 1.9 },
    { t: "couch",    l: 1.6 },
    { t: "lap",      l: 1.1 },
    { t: "roof",     l: 0.4 },
    { t: "keyboard", l: -0.3 },
    { t: "moon",     l: -1.2 }
  ];

  var tEl = document.getElementById("temp"),
      kEl = document.getElementById("topk"),
      pEl = document.getElementById("topp"),
      tVal = document.getElementById("temp-val"),
      kVal = document.getElementById("topk-val"),
      pVal = document.getElementById("topp-val"),
      sampleBtn = document.getElementById("sample-btn"),
      sampleOut = document.getElementById("sample-out");

  function softmax(logits, T) {
    var scaled = logits.map(function (l) { return l / T; });
    var m = Math.max.apply(null, scaled);
    var ex = scaled.map(function (v) { return Math.exp(v - m); });
    var sum = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (v) { return v / sum; });
  }

  // Returns { base, survive[], finalP[] } for the current control values.
  function compute() {
    var T = parseFloat(tEl.value), K = parseInt(kEl.value, 10), P = parseFloat(pEl.value);
    var base = softmax(TOKENS.map(function (d) { return d.l; }), T);

    // rank indices by probability, high → low
    var order = base.map(function (_, i) { return i; })
                    .sort(function (a, b) { return base[b] - base[a]; });

    var inK = {};
    order.slice(0, K).forEach(function (i) { inK[i] = true; });

    var inP = {}, cum = 0;
    for (var r = 0; r < order.length; r++) {
      var idx = order[r];
      inP[idx] = true;          // always include at least the top token
      cum += base[idx];
      if (cum >= P) break;
    }

    var survive = TOKENS.map(function (_, i) { return !!(inK[i] && inP[i]); });
    var surviveSum = base.reduce(function (a, p, i) { return a + (survive[i] ? p : 0); }, 0);
    var finalP = base.map(function (p, i) { return survive[i] ? p / surviveSum : 0; });
    return { base: base, survive: survive, finalP: finalP };
  }

  var state;

  function render() {
    state = compute();
    var maxBase = Math.max.apply(null, state.base);
    dist.innerHTML = "";
    TOKENS.forEach(function (d, i) {
      var row = document.createElement("div");
      row.className = "tok-row" + (state.survive[i] ? "" : " cut");
      row.dataset.idx = String(i);
      var width = (state.base[i] / maxBase * 100).toFixed(1);
      var pct = (state.base[i] * 100).toFixed(1);
      row.innerHTML =
        '<span class="tok">' + d.t + '</span>' +
        '<span class="track"><span class="bar" style="width:' + width + '%"></span></span>' +
        '<span class="pct">' + pct + '%</span>';
      dist.appendChild(row);
    });
    tVal.textContent = parseFloat(tEl.value).toFixed(2);
    kVal.textContent = kEl.value;
    pVal.textContent = parseFloat(pEl.value).toFixed(2);
    sampleOut.textContent = "";
  }

  function sample() {
    if (!state) state = compute();
    var rnd = Math.random(), cum = 0, pick = -1;
    for (var i = 0; i < TOKENS.length; i++) {
      cum += state.finalP[i];
      if (rnd <= cum) { pick = i; break; }
    }
    if (pick < 0) {                       // floating-point guard: take the last survivor
      for (var j = TOKENS.length - 1; j >= 0; j--) { if (state.survive[j]) { pick = j; break; } }
    }
    Array.prototype.forEach.call(dist.children, function (c) {
      c.classList.toggle("sampled", parseInt(c.dataset.idx, 10) === pick);
    });
    var blank = document.getElementById("blank");
    if (blank) blank.textContent = TOKENS[pick].t;
    sampleOut.innerHTML = 'Sampled: <strong>“' + TOKENS[pick].t + '”</strong>';
  }

  [tEl, kEl, pEl].forEach(function (el) { el.addEventListener("input", render); });
  sampleBtn.addEventListener("click", sample);
  render();
})();
