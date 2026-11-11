// Real-browser check of docs/build.html (rules R10, R14) in three modes: desktop light, dark, phone 400 px.
// Every phase opens from the picker and the system map; every swimlane step shows a tooltip on hover and
// hides it on leave; Next/Prev walk all steps and update the detail panel; no tooltip on load; no sideways
// page scroll; no JS errors; every theme colour token resolves. Run: make build-test
const puppeteer = require("puppeteer-core");
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const [file, outDir] = process.argv.slice(2);
  const b = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", args: (process.env.CHROME_ARGS || "").split(" ").filter(Boolean), headless: "new" });
  const modes = [["desktop-light", 1280, 900, "light"], ["desktop-dark", 1280, 900, "dark"], ["phone-400", 400, 860, "light"]];
  let fails = 0, checks = 0;
  const ok = (cond, msg) => { checks++; if (!cond) { fails++; console.log("FAIL", msg); } };
  for (const [mode, W, H, scheme] of modes) {
    const page = await b.newPage(); await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: scheme }]);
    const errs = []; page.on("pageerror", (e) => errs.push(e.message));
    await page.goto("file://" + file, { waitUntil: "networkidle0" });
    ok(await page.$eval("#tip", (t) => t.hidden), `${mode}: tooltip hidden on load`);
    const phases = await page.$$eval("#picker button", (bs) => bs.map((x) => x.dataset.id));
    const nPhases = await page.evaluate("PHASES.length"); ok(phases.length === nPhases && nPhases > 0, `${mode}: ${nPhases} phases`);
    for (const id of phases) {
      await page.click(`#picker button[data-id="${id}"]`); await wait(60);
      ok(await page.$eval(`#picker button[data-id="${id}"]`, (x) => x.getAttribute("aria-selected") === "true"), `${mode} ${id}: selected`);
      const n = await page.$$eval("#swim g.step", (gs) => gs.length);
      ok(n >= 3, `${mode} ${id}: has steps (${n})`);
      for (let i = 0; i < n; i++) {
        const g = await page.$(`#swim g.step[data-i="${i}"]`); await g.scrollIntoView(); await wait(150);   // let scroll events settle: they hide the tooltip by design
        const bb = await g.boundingBox(); await page.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2); await wait(40);
        const t = await page.$eval("#tip", (t) => ({ shown: !t.hidden, text: t.textContent.trim() }));
        ok(t.shown && t.text.startsWith(`${i + 1}.`), `${mode} ${id}: step ${i + 1} tooltip (${t.text.slice(0, 30)})`);
        await page.mouse.move(1, 1); await page.evaluate(() => window.dispatchEvent(new Event("scroll"))); await wait(30);
      }
      ok(await page.$eval("#tip", (t) => t.hidden), `${mode} ${id}: tooltip hides on leave`);
      for (let i = 1; i < n; i++) { await page.click("#next"); await wait(20); }
      const end = await page.evaluate(() => ({ pos: document.querySelector("#pos").textContent, dis: document.querySelector("#next").disabled, cur: document.querySelector("#swim g.step.current").dataset.i }));
      ok(end.pos === `Step ${n} of ${n}` && end.dis && +end.cur === n - 1, `${mode} ${id}: Next walks to the last step (${end.pos})`);
      await page.click("#prev"); await wait(20);
      ok(await page.$eval("#detail .eyebrow", (e, n) => e.textContent.startsWith(`Step ${n - 1} of`), n), `${mode} ${id}: Prev updates detail`);
      ok(await page.$$eval("#phase .scqa > div", (d) => d.length === 4), `${mode} ${id}: SCQA strip`);
    }
    const cell = await page.$('#map td.cell[data-id="p3"]'); await cell.scrollIntoView(); await cell.click(); await wait(400);
    ok(await page.$eval('#picker button[data-id="p3"]', (x) => x.getAttribute("aria-selected") === "true"), `${mode}: map opens a phase`);
    const hs = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    ok(!hs, `${mode}: no sideways page scroll`);
    const colors = await page.evaluate(() => { const s = getComputedStyle(document.documentElement); return ["--bg", "--fg", "--accent", "--surface"].every((v) => s.getPropertyValue(v).trim()); });
    ok(colors, `${mode}: colour tokens resolve`);
    await page.click('#picker button[data-id="p6"]'); await wait(80);
    await page.screenshot({ path: `${outDir}/build_${mode}.png`, fullPage: false });
    ok(errs.length === 0, `${mode}: no JS errors ${errs.join("; ")}`);
    await page.close();
  }
  console.log(`${checks - fails}/${checks} build-page checks pass`);
  await b.close();
  process.exit(fails ? 1 : 0);
})();
