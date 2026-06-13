/* Failed Star — diagrams.js
   Interactive previews. No framework, no build step. Each explorer is guarded by
   its root element, so the file is harmless to load on any page. */

(function () {
  "use strict";

  initSampler();
  initAttention();

  /* === Sampling: real softmax over toy logits, reshaped by temp / top-k / top-p === */
  function initSampler() {
    var dist = document.getElementById("dist");
    if (!dist) return;

    var TOKENS = [
      { t: "mat", l: 3.1 }, { t: "floor", l: 2.4 }, { t: "rug", l: 1.9 },
      { t: "couch", l: 1.6 }, { t: "lap", l: 1.1 }, { t: "roof", l: 0.4 },
      { t: "keyboard", l: -0.3 }, { t: "moon", l: -1.2 }
    ];
    var tEl = document.getElementById("temp"), kEl = document.getElementById("topk"),
        pEl = document.getElementById("topp"), tVal = document.getElementById("temp-val"),
        kVal = document.getElementById("topk-val"), pVal = document.getElementById("topp-val"),
        sampleBtn = document.getElementById("sample-btn"),
        sampleOut = document.getElementById("sample-out");
    var state;

    function compute() {
      var T = parseFloat(tEl.value), K = parseInt(kEl.value, 10), P = parseFloat(pEl.value);
      var base = softmax(TOKENS.map(function (d) { return d.l / T; }));
      var order = base.map(function (_, i) { return i; })
                      .sort(function (a, b) { return base[b] - base[a]; });
      var inK = {}; order.slice(0, K).forEach(function (i) { inK[i] = true; });
      var inP = {}, cum = 0;
      for (var r = 0; r < order.length; r++) {
        var idx = order[r]; inP[idx] = true; cum += base[idx]; if (cum >= P) break;
      }
      var survive = TOKENS.map(function (_, i) { return !!(inK[i] && inP[i]); });
      var sum = base.reduce(function (a, p, i) { return a + (survive[i] ? p : 0); }, 0);
      var finalP = base.map(function (p, i) { return survive[i] ? p / sum : 0; });
      return { base: base, survive: survive, finalP: finalP };
    }

    function render() {
      state = compute();
      var maxBase = Math.max.apply(null, state.base);
      dist.innerHTML = "";
      TOKENS.forEach(function (d, i) {
        var row = document.createElement("div");
        row.className = "tok-row" + (state.survive[i] ? "" : " cut");
        row.dataset.idx = String(i);
        row.innerHTML =
          '<span class="tok">' + d.t + '</span>' +
          '<span class="track"><span class="bar" style="width:' +
            (state.base[i] / maxBase * 100).toFixed(1) + '%"></span></span>' +
          '<span class="pct">' + (state.base[i] * 100).toFixed(1) + '%</span>';
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
        cum += state.finalP[i]; if (rnd <= cum) { pick = i; break; }
      }
      if (pick < 0) {                       // floating-point guard: last survivor
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
  }

  /* === Attention: real scaled-dot-product softmax over toy token vectors === */
  function initAttention() {
    var tokensEl = document.getElementById("attn-tokens");
    if (!tokensEl) return;

    var TOKENS = ["The", "cat", "sat", "on", "the", "mat"];
    var EMB = [
      [1, 0, 0], [0.2, 1, 0], [0, 0.8, 0.6], [0.1, 0.1, 1], [0.9, 0.1, 0.2], [0, 0.7, 0.7]
    ];
    var SCALE = Math.sqrt(EMB[0].length);
    var barsEl = document.getElementById("attn-bars"),
        capEl = document.getElementById("attn-caption"),
        causalEl = document.getElementById("attn-causal");
    var query = TOKENS.length - 1;

    function render() {
      var hi = causalEl.checked ? query : TOKENS.length - 1;
      var idxs = []; for (var j = 0; j <= hi; j++) idxs.push(j);
      var w = softmax(idxs.map(function (j) { return dot(EMB[query], EMB[j]) / SCALE; }));

      tokensEl.innerHTML = "";
      TOKENS.forEach(function (t, j) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "attn-chip" + (j === query ? " query" : "");
        chip.dataset.j = String(j);
        chip.appendChild(document.createTextNode(t));
        var pos = idxs.indexOf(j);
        if (pos >= 0) {
          var bar = document.createElement("span");
          bar.className = "attn-wbar";
          bar.style.width = (w[pos] * 100).toFixed(1) + "%";
          chip.appendChild(bar);
        } else {
          chip.classList.add("masked");
        }
        chip.addEventListener("click", function () {
          query = parseInt(this.dataset.j, 10); render();
        });
        tokensEl.appendChild(chip);
      });

      var maxw = Math.max.apply(null, w);
      barsEl.innerHTML = "";
      idxs.forEach(function (j, pos) {
        var row = document.createElement("div");
        row.className = "tok-row";
        row.innerHTML =
          '<span class="tok">' + TOKENS[j] + '</span>' +
          '<span class="track"><span class="bar" style="width:' +
            (w[pos] / maxw * 100).toFixed(1) + '%"></span></span>' +
          '<span class="pct">' + (w[pos] * 100).toFixed(0) + '%</span>';
        barsEl.appendChild(row);
      });

      var ranked = idxs.map(function (j, pos) { return { j: j, w: w[pos] }; })
                       .sort(function (a, b) { return b.w - a.w; });
      var top = ranked.slice(0, 2).map(function (r) { return "“" + TOKENS[r.j] + "”"; });
      capEl.innerHTML = "<strong>“" + TOKENS[query] + "”</strong> attends most to " +
                        top.join(" and ") + ".";
    }

    causalEl.addEventListener("change", render);
    render();
  }

  /* === shared helpers === */
  function softmax(xs) {
    var m = Math.max.apply(null, xs);
    var ex = xs.map(function (v) { return Math.exp(v - m); });
    var sum = ex.reduce(function (a, b) { return a + b; }, 0);
    return ex.map(function (v) { return v / sum; });
  }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
})();
