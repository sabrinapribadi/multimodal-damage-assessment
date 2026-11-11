// Real-browser check of a results page made from the shared template (rules R10, R12): desktop light, dark, phone 400 px.
// Nothing open or shown on load; every chart mark shows its tooltip on hover and hides it on leave; sliders redraw;
// no sideways page scroll; no JS errors. Run: node tests/browser/hover_test.js <docs/index.html>
const puppeteer = require("puppeteer-core");
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const [file] = process.argv.slice(2);
  const b = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    args: (process.env.CHROME_ARGS || "").split(" ").filter(Boolean),
  });
  let fails = 0, checks = 0;
  const ok = (c, m) => { checks++; if (!c) { fails++; console.log("FAIL", m); } };
  for (const [mode, W, H, scheme] of [["desktop-light", 1280, 900, "light"], ["desktop-dark", 1280, 900, "dark"], ["phone-400", 400, 860, "light"]]) {
    const page = await b.newPage(); await page.setViewport({ width: W, height: H });
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: scheme }]);
    const errs = []; page.on("pageerror", (e) => errs.push(e.message));
    await page.goto("file://" + file, { waitUntil: "networkidle0" });
    ok(await page.$eval("#tip", (t) => t.hidden), `${mode}: tooltip hidden on load`);
    ok(await page.$$eval("details.more", (ds) => ds.length > 0 && ds.every((d) => !d.open)), `${mode}: collapsible sections closed on load`);
    ok((await page.$("#brief")) !== null, `${mode}: brief present`);
    await page.$$eval("details", (ds) => ds.forEach((d) => (d.open = true)));
    const n = await page.$$eval(".chart [data-tip]", (els) => els.length);
    ok(n > 0, `${mode}: chart marks found (${n})`);
    for (let i = 0; i < n; i++) {
      const el = (await page.$$(".chart [data-tip]"))[i];
      await el.scrollIntoView(); await wait(30);
      const bb = await el.boundingBox();
      if (!bb) { ok(false, `${mode}: mark ${i} has no box`); continue; }
      await page.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2);
      const t = await page.$eval("#tip", (t) => ({ shown: !t.hidden, text: t.textContent.trim() }));
      ok(t.shown && t.text.length > 5, `${mode}: mark ${i} tooltip (${t.text.slice(0, 30)})`);
      await page.mouse.move(1, 1); await page.evaluate(() => window.dispatchEvent(new Event("scroll")));
    }
    ok(await page.$eval("#tip", (t) => t.hidden), `${mode}: tooltip hides on leave`);
    for (const s of await page.$$(".slider input[type=range]")) {
      const before = await s.evaluate((el) => el.parentElement.querySelector("span").textContent);
      await s.evaluate((el) => { el.value = String(+el.value > 0 ? +el.value - 1 : +el.value + 1); el.dispatchEvent(new Event("input")); });
      const after = await s.evaluate((el) => el.parentElement.querySelector("span").textContent);
      ok(before !== after, `${mode}: slider redraws (${before} → ${after})`);
    }
    ok(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)), `${mode}: no sideways scroll`);
    ok(errs.length === 0, `${mode}: no JS errors (${errs.join("; ")})`);
    await page.close();
  }
  console.log(`${checks - fails}/${checks} results-page checks pass`);
  await b.close();
  process.exit(fails ? 1 : 0);
})();
