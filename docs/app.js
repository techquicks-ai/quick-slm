/* Quick SLM documentation site: theme, sidebar, active-section tracking.
  Shared by index.html, and . */
(function () {
 var root = document.documentElement;

 // ---- theme: auto -> light -> dark, remembered ----
 var btn = document.getElementById('themeToggle');
 var label = document.getElementById('themeLabel');
 var order = ['auto', 'light', 'dark'];
 var names = { auto: 'Auto', light: 'Light', dark: 'Dark' };

 function apply(mode) {
  if (mode === 'auto') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', mode);
  if (label) label.textContent = names[mode];
  if (btn) btn.dataset.mode = mode;
  try { localStorage.setItem('qslm-theme', mode); } catch (e) {}
 }
 var saved = null;
 try { saved = localStorage.getItem('qslm-theme'); } catch (e) {}
 apply(saved && order.indexOf(saved) >= 0 ? saved : 'auto');

 if (btn) {
  btn.addEventListener('click', function () {
   apply(order[(order.indexOf(btn.dataset.mode || 'auto') + 1) % order.length]);
  });
 }

 // ---- sidebar, off-canvas below 900px ----
 var sidebar = document.querySelector('.sidebar');
 var menu = document.getElementById('menuToggle');
 if (menu && sidebar) {
  menu.addEventListener('click', function (e) {
   e.stopPropagation();
   sidebar.classList.toggle('open');
   menu.setAttribute('aria-expanded', sidebar.classList.contains('open') ? 'true' : 'false');
  });
  // A tap on a link, or anywhere outside, closes it again.
  sidebar.addEventListener('click', function (e) {
   if (e.target.tagName === 'A') sidebar.classList.remove('open');
  });
  document.addEventListener('click', function (e) {
   if (!sidebar.contains(e.target) && e.target !== menu) sidebar.classList.remove('open');
  });
  document.addEventListener('keydown', function (e) {
   if (e.key === 'Escape') sidebar.classList.remove('open');
  });
 }

 // ---- highlight the section currently in view ----
 var links = Array.prototype.slice.call(document.querySelectorAll('.sidebar .sections a'));
 if (!links.length) return;
 var byId = {};
 links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });

 var obs = new IntersectionObserver(function (entries) {
  entries.forEach(function (en) {
   if (!en.isIntersecting) return;
   links.forEach(function (l) { l.classList.remove('active'); });
   if (byId[en.target.id]) byId[en.target.id].classList.add('active');
  });
 }, { rootMargin: '-40% 0px -55% 0px' });

 document.querySelectorAll('main section[id]').forEach(function (s) { obs.observe(s); });
})();
