/* Failed Star — tiny vanilla JS. No framework, no build step.
   (1) "Show what we build" toggle highlights the rungs/stages we implement.
   (2) Accordion niceness: opening one rung is fine; we don't force-close others,
       but we do scroll a freshly opened rung into comfortable view on mobile. */

(function () {
  "use strict";

  // (1) Build/cite highlight toggle on the ladder.
  var ladder = document.querySelector(".ladder");
  var toggle = document.getElementById("show-build");
  if (ladder && toggle) {
    var apply = function () { ladder.classList.toggle("show-build", toggle.checked); };
    toggle.addEventListener("change", apply);
    apply();
  }

  // (2) On small screens, scroll an opened rung into view so its detail is visible.
  var rungs = document.querySelectorAll(".rung");
  rungs.forEach(function (rung) {
    rung.addEventListener("toggle", function () {
      if (rung.open && window.matchMedia("(max-width: 760px)").matches) {
        rung.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });
  });

  // Stamp the build year in the footer (kept out of HTML so it never goes stale).
  var y = document.getElementById("year");
  if (y) { y.textContent = String(new Date().getFullYear()); }
})();
