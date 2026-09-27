// Optional browser checks; CI installs the pinned Playwright version.
const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {chromium} = require(process.env.TREX_PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const site = path.join(__dirname, '_site');
  const routes = fs.readdirSync(site, {recursive: true})
    .filter(p => p.endsWith('.html') && !p.startsWith('site_libs/'));
  const base = process.env.TREX_DOCS_URL || 'http://127.0.0.1:8916/';
  let server;
  let browser;
  try {
    if (!process.env.TREX_DOCS_URL) {
      server = spawn('python3', ['-m', 'http.server', '8916', '--bind', '127.0.0.1', '--directory', site], {stdio: 'ignore'});
      let ready = false;
      for (let i = 0; i < 30; i++) {
        try { if ((await fetch(base)).ok) { ready = true; break; } } catch {}
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      if (!ready) throw new Error('Local preview server did not start');
    }
    browser = await chromium.launch({headless: true, ...(process.env.TREX_CHROME_CHANNEL ? {channel: process.env.TREX_CHROME_CHANNEL} : {})});
    const failures = [];
    let inspected = 0;
    for (const width of [1440, 390]) {
      const context = await browser.newContext({viewport: {width, height: 900}});
      const queue = [...routes];
      await Promise.all(Array.from({length: 4}, async () => {
        const page = await context.newPage();
        let current;
        page.on('pageerror', e => failures.push(`${current}: ${e.message}`));
        page.on('request', r => {
          if (/^https?:/.test(r.url()) && new URL(r.url()).origin !== new URL(base).origin)
            failures.push(`${current}: external runtime request ${r.url()}`);
        });
        while (queue.length) {
          const route = queue.shift();
          current = `${width}px ${route}`;
          const response = await page.goto(new URL(route, base).href, {waitUntil: 'networkidle'});
          if (!response.ok()) failures.push(`${current}: HTTP ${response.status()}`);
          const state = await page.evaluate(() => ({
            width: document.documentElement.scrollWidth, viewport: innerWidth,
            math: document.querySelectorAll('.math').length,
            renderedMath: document.querySelectorAll('.math .katex').length,
            mathErrors: document.querySelectorAll('.katex-error').length,
            brokenImages: [...document.images].filter(im => !im.complete || im.naturalWidth === 0).map(im => im.src),
          }));
          if (state.width > state.viewport + 2) failures.push(`${current}: horizontal overflow ${state.width}`);
          if (state.math !== state.renderedMath || state.mathErrors) failures.push(`${current}: math ${JSON.stringify(state)}`);
          if (state.brokenImages.length) failures.push(`${current}: broken images ${state.brokenImages}`);
          inspected++;
        }
        await page.close();
      }));
      await context.close();
    }
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    await page.goto(base, {waitUntil: 'networkidle'});
    await page.locator('#quarto-search').click();
    await page.locator('input.aa-Input').fill('HotzMiller');
    await page.waitForFunction(() => document.querySelectorAll('.aa-Item').length > 0);
    const searchText = await page.locator('.aa-Panel').innerText();
    if (!/Hotz|dynamic/i.test(searchText)) failures.push('Search did not return the dynamic-choice material');
    await page.keyboard.press('Escape');
    await page.locator('.quarto-color-scheme-toggle').first().click();
    await page.waitForFunction(() => document.body.classList.contains('quarto-dark'));
    await page.locator('.quarto-color-scheme-toggle').first().click();
    const screenshotDir = process.env.TREX_DOCS_SCREENSHOTS;
    if (screenshotDir) {
      fs.mkdirSync(screenshotDir, {recursive: true});
      await page.screenshot({path: path.join(screenshotDir, 'desktop.png'), fullPage: true});
    }
    await page.setViewportSize({width: 390, height: 844});
    await page.goto(new URL('guides/synthetic-did.html', base).href, {waitUntil: 'networkidle'});
    await page.getByRole('button', {name: 'Toggle sidebar navigation'}).click();
    await page.waitForFunction(() => document.querySelector('#quarto-sidebar').classList.contains('show'));
    const menu = await page.evaluate(() => {
      const el = document.querySelector('#quarto-sidebar');
      return {right: el.getBoundingClientRect().right, viewport: innerWidth,
              scrollWidth: el.scrollWidth, clientWidth: el.clientWidth};
    });
    if (menu.right > menu.viewport + 2 || menu.scrollWidth > menu.clientWidth + 2)
      failures.push(`Mobile navigation overflow: ${JSON.stringify(menu)}`);
    await page.getByRole('button', {name: 'Toggle sidebar navigation'}).click();
    await page.waitForFunction(() => document.querySelector('#quarto-sidebar').getBoundingClientRect().width === 0);
    if (screenshotDir) await page.screenshot({path: path.join(screenshotDir, 'mobile.png'), fullPage: true});
    if (failures.length) throw new Error([...new Set(failures)].join('\n'));
    console.log(JSON.stringify({page_viewports_checked: inspected, widths: [1440, 390], search: 'passed', mobile_navigation: 'passed', theme_toggle: 'passed'}));
  } finally {
    if (browser) await browser.close();
    if (server) server.kill();
  }
}
main().catch(e => {console.error(e); process.exitCode = 1;});
