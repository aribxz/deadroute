const fs = require("node:fs");
const path = require("node:path");

const projectRoot = path.resolve(__dirname, "..");
const frontendDir = path.join(projectRoot, "frontend");
const outputDir = path.join(projectRoot, "netlify-build");
const configuredApi = (process.env.DEADROUTE_API_BASE || "").trim();

if (!configuredApi) {
  throw new Error(
    "Set DEADROUTE_API_BASE in Netlify's environment variables to the public HTTPS root URL of a working DeadRoute API."
  );
}

let apiUrl;
try {
  apiUrl = new URL(configuredApi);
} catch {
  throw new Error("DEADROUTE_API_BASE must be a valid HTTPS URL, such as https://deadroute-api.onrender.com.");
}
if (apiUrl.protocol !== "https:" || apiUrl.username || apiUrl.password || apiUrl.search || apiUrl.hash || (apiUrl.pathname !== "/" && apiUrl.pathname !== "")) {
  throw new Error("DEADROUTE_API_BASE must be an HTTPS origin with no path, credentials, query, or fragment.");
}

fs.rmSync(outputDir, { recursive: true, force: true });
fs.cpSync(frontendDir, outputDir, { recursive: true });
fs.writeFileSync(
  path.join(outputDir, "config.js"),
  'window.DEADROUTE_API_BASE = "/api";\n',
  "utf8"
);
fs.writeFileSync(
  path.join(outputDir, "_redirects"),
  `/api/* ${apiUrl.origin}/:splat 200\n`,
  "utf8"
);
console.log(`Netlify frontend built in ${path.relative(projectRoot, outputDir)}; API proxy targets ${apiUrl.origin}.`);
