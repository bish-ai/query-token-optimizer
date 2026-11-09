"use strict";

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const PORT = Number(process.env.PORT) || 3000;
const PUBLIC_DIR = path.join(__dirname, "public");
const CATALOG_PATH = path.join(__dirname, "data", "models.json");
const MAX_BODY_BYTES = 8 * 1024;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml"
};

function sendJson(res, status, value) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff"
  });
  res.end(JSON.stringify(value));
}

function readCatalog() {
  return JSON.parse(fs.readFileSync(CATALOG_PATH, "utf8"));
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    let size = 0;

    req.on("data", chunk => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("Request body is too large."));
        req.resume();
        return;
      }
      body += chunk;
    });

    req.on("end", () => {
      try {
        resolve(JSON.parse(body || "{}"));
      } catch {
        reject(new Error("Request body must be valid JSON."));
      }
    });

    req.on("error", reject);
  });
}

function estimateCost(model, inputTokens, outputTokens) {
  return (
    inputTokens * model.inputPerMillion +
    outputTokens * model.outputPerMillion
  ) / 1_000_000;
}

async function serveStatic(pathname, res) {
  let decodedPath;
  try {
    decodedPath = decodeURIComponent(pathname);
  } catch {
    sendJson(res, 400, { error: "Invalid URL path." });
    return;
  }

  const relativePath = decodedPath === "/" ? "index.html" : decodedPath.slice(1);
  const filePath = path.resolve(PUBLIC_DIR, relativePath);
  const relativeToPublic = path.relative(PUBLIC_DIR, filePath);

  if (relativeToPublic.startsWith("..") || path.isAbsolute(relativeToPublic)) {
    sendJson(res, 403, { error: "Forbidden." });
    return;
  }

  try {
    const content = await fs.promises.readFile(filePath);
    res.writeHead(200, {
      "Content-Type": MIME_TYPES[path.extname(filePath)] || "application/octet-stream",
      "X-Content-Type-Options": "nosniff"
    });
    res.end(content);
  } catch {
    sendJson(res, 404, { error: "File not found." });
  }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);

  try {
    if (req.method === "GET" && url.pathname === "/api/models") {
      const catalog = readCatalog();
      sendJson(res, 200, {
        useCases: catalog.useCases,
        modelCount: catalog.models.length
      });
      return;
    }

    if (req.method === "POST" && url.pathname === "/api/quote") {
      const body = await readJsonBody(req);
      const inputTokens = body.inputTokens;
      const outputTokens = body.outputTokens;
      const useCase = body.useCase;
      const catalog = readCatalog();

      const validUseCase = catalog.useCases.some(item => item.id === useCase);
      const validCounts =
        Number.isSafeInteger(inputTokens) &&
        Number.isSafeInteger(outputTokens) &&
        inputTokens >= 0 && inputTokens <= 2_000_000 &&
        outputTokens >= 1 && outputTokens <= 100_000;

      if (!validUseCase || !validCounts) {
        sendJson(res, 400, { error: "Invalid use case or token estimate." });
        return;
      }

      const matchingModels = catalog.models
        .filter(model => model.useCases.includes(useCase))
        .map(model => ({
          id: model.id,
          name: model.name,
          provider: model.provider,
          inputPerMillion: model.inputPerMillion,
          outputPerMillion: model.outputPerMillion,
          estimatedCost: estimateCost(model, inputTokens, outputTokens)
        }))
        .sort((a, b) => a.estimatedCost - b.estimatedCost);

      sendJson(res, 200, {
        useCase,
        inputTokens,
        outputTokens,
        recommendation: matchingModels[0] || null,
        models: matchingModels
      });
      return;
    }

    if (url.pathname.startsWith("/api/")) {
      sendJson(res, 404, { error: "API endpoint not found." });
      return;
    }

    if (req.method !== "GET" && req.method !== "HEAD") {
      sendJson(res, 405, { error: "Method not allowed." });
      return;
    }

    await serveStatic(url.pathname, res);
  } catch (error) {
    sendJson(res, error.message === "Request body is too large." ? 413 : 400, {
      error: error.message || "Request failed."
    });
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`QueryWise running at http://localhost:${PORT}`);
});
