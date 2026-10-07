(function () {
  var root = document.documentElement;

  // Theme toggle: explicit choice is stored, otherwise follow the OS setting.
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      var next = current === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try {
        localStorage.setItem("theme", next);
      } catch (e) {}
    });
  }

  var prose = document.querySelector(".prose");
  if (!prose) return;

  // Table of contents built from the article's top-level headings.
  var toc = document.querySelector(".toc");
  var headings = Array.prototype.filter.call(prose.querySelectorAll("h1[id], h2[id]"), function (h) {
    return h.textContent.trim();
  });
  if (toc && headings.length > 2) {
    var topLevel = prose.querySelector("h1[id]") ? "H1" : "H2";
    var list = toc.querySelector("ol");
    var links = headings.map(function (h) {
      var li = document.createElement("li");
      if (h.tagName !== topLevel) li.className = "sub";
      var a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent.trim();
      li.appendChild(a);
      list.appendChild(li);
      return a;
    });
    toc.hidden = false;

    var setActive = function () {
      var idx = 0;
      for (var i = 0; i < headings.length; i++) {
        if (headings[i].getBoundingClientRect().top < 120) idx = i;
      }
      links.forEach(function (a, i) {
        a.classList.toggle("active", i === idx);
      });
    };
    addEventListener("scroll", setActive, { passive: true });
    setActive();
  }

  // Copy buttons on code blocks.
  prose.querySelectorAll("div.highlighter-rouge").forEach(function (block) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.addEventListener("click", function () {
      var code = block.querySelector("code");
      navigator.clipboard.writeText(code ? code.innerText : "").then(function () {
        btn.textContent = "Copied";
        setTimeout(function () {
          btn.textContent = "Copy";
        }, 1500);
      });
    });
    block.appendChild(btn);
  });

  // Click-to-zoom figures (medium-zoom is loaded on post pages).
  addEventListener("load", function () {
    if (window.mediumZoom) window.mediumZoom("[data-zoomable]", { margin: 24 });
  });
})();
