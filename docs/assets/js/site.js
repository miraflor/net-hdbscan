(() => {
  const body = document.body;
  const menu = document.getElementById('mobile-menu');
  const scrim = document.getElementById('sidebar-scrim');
  const search = document.getElementById('doc-search');
  const navLinks = [...document.querySelectorAll('.doc-nav a[href^="#"]')];
  const content = document.getElementById('doc-content');
  const topButton = document.getElementById('back-to-top');

  const closeNav = () => {
    body.classList.remove('nav-open');
    menu?.setAttribute('aria-expanded', 'false');
  };

  menu?.addEventListener('click', () => {
    const open = body.classList.toggle('nav-open');
    menu.setAttribute('aria-expanded', String(open));
  });

  scrim?.addEventListener('click', closeNav);
  navLinks.forEach(link => link.addEventListener('click', closeNav));

  content.querySelectorAll('h1[id], h2[id], h3[id], h4[id]').forEach(h => {
    const a = document.createElement('a');
    a.href = `#${h.id}`;
    a.className = 'headerlink';
    a.setAttribute('aria-label', `Permalink to ${h.textContent}`);
    a.textContent = '¶';
    h.appendChild(a);
  });

  content.querySelectorAll('pre').forEach(pre => {
    const code = pre.querySelector('code');
    if (!code) return;

    if (code.classList.contains('language-mermaid')) {
      const diagram = document.createElement('div');
      diagram.className = 'mermaid';
      diagram.textContent = code.textContent;
      pre.replaceWith(diagram);
      return;
    }

    const button = document.createElement('button');
    button.className = 'copy-code';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.innerText);
        button.textContent = 'Copied';
        setTimeout(() => button.textContent = 'Copy', 1200);
      } catch {
        button.textContent = 'Select';
      }
    });
    pre.appendChild(button);
  });

  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: false,
      theme: 'neutral',
      securityLevel: 'strict'
    });
    window.mermaid.run({ querySelector: '.mermaid' });
  }

  const applySearch = () => {
    const q = search.value.trim().toLowerCase();
    navLinks.forEach(link => {
      const label = (link.dataset.navLabel || link.textContent).toLowerCase();
      link.classList.toggle('search-hidden', q && !label.includes(q));
    });
  };

  search?.addEventListener('input', applySearch);

  document.addEventListener('keydown', event => {
    if (event.key === '/' &&
        document.activeElement !== search &&
        !['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
      event.preventDefault();
      search?.focus();
    }

    if (event.key === 'Escape' && document.activeElement === search) {
      search.value = '';
      applySearch();
      search.blur();
    }
  });

  const sections = navLinks.map(link => {
    try {
      return { link, el: document.querySelector(link.getAttribute('href')) };
    } catch {
      return null;
    }
  }).filter(x => x && x.el);

  const setActive = () => {
    let current = sections[0];
    const y = window.scrollY + 90;

    for (const section of sections) {
      if (section.el.offsetTop <= y) current = section;
      else break;
    }

    sections.forEach(section => {
      section.link.classList.toggle('active', section === current);
    });

    topButton?.classList.toggle('visible', window.scrollY > 500);
  };

  window.addEventListener('scroll', setActive, { passive: true });
  window.addEventListener('resize', setActive);
  setActive();

  topButton?.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
})();
