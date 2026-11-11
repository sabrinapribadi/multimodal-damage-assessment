// Share-preview image (portfolio rule R18): the top 1200 x 627 of a page, light mode, at 2x then saved as PNG.
// Run: make preview   (needs Node + Google Chrome)   args: <page.html> <out.png>
const puppeteer = require("puppeteer-core");
(async () => {
  const [file, out] = process.argv.slice(2);
  const b = await puppeteer.launch({ executablePath: process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", args: (process.env.CHROME_ARGS || "").split(" ").filter(Boolean), headless: "new" });
  const page = await b.newPage();
  await page.setViewport({ width: 1200, height: 627, deviceScaleFactor: 1 });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
  await page.goto("file://" + file, { waitUntil: "networkidle0", timeout: 60000 });
  await new Promise((r) => setTimeout(r, 800));
  await page.screenshot({ path: out, clip: { x: 0, y: 0, width: 1200, height: 627 } });
  console.log("preview", out);
  await b.close();
})();
