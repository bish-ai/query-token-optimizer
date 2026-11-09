import("https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js")
  .then(({ WebGLRenderer, PerspectiveCamera, Scene, Mesh, SphereGeometry, MeshBasicMaterial, Color, Group }) => {
    const canvas = document.getElementById("scene");
    const renderer = new WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const scene = new Scene();
    const camera = new PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.z = 7;

    const group = new Group();
    scene.add(group);

    const particles = [
      { position: [-2.7, 0.8, 0], radius: 1.05, color: "#9cb0ff" },
      { position: [2.4, -0.5, -0.4], radius: 0.85, color: "#79e0c0" },
      { position: [0.2, 1.6, -0.8], radius: 1.25, color: "#b3c1ff" }
    ].map(({ position, radius, color }) => {
      const material = new MeshBasicMaterial({
        color: new Color(color),
        wireframe: true,
        transparent: true,
        opacity: 0.42
      });
      const mesh = new Mesh(new SphereGeometry(radius, 24, 24), material);
      mesh.position.set(...position);
      group.add(mesh);
      return mesh;
    });

    const resize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };

    resize();
    window.addEventListener("resize", resize);

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame = 0;
    const animate = () => {
      if (!reduceMotion) {
        frame += 0.003;
        particles.forEach((mesh, index) => {
          mesh.rotation.x += 0.0007 * (index + 1);
          mesh.rotation.y += 0.001 * (index + 1);
        });
        group.rotation.z = Math.sin(frame) * 0.08;
      }
      renderer.render(scene, camera);
      if (!reduceMotion) requestAnimationFrame(animate);
    };
    animate();
  })
  .catch(() => {
    // The recommender remains functional if WebGL or the Three.js CDN is unavailable.
  });

const elements = {
  query: document.getElementById("query"),
  tokenEstimate: document.getElementById("token-estimate"),
  useCase: document.getElementById("useCase"),
  outputTokens: document.getElementById("outputTokens"),
  button: document.getElementById("recommendButton"),
  error: document.getElementById("error"),
  results: document.getElementById("results"),
  task: document.getElementById("result-task"),
  recommendation: document.getElementById("recommendation"),
  modelList: document.getElementById("model-list")
};

document.getElementById("year").textContent = new Date().getFullYear();

function estimateTokens(text) {
  const trimmed = text.trim();
  return trimmed ? Math.max(1, Math.ceil(trimmed.length / 4)) : 0;
}

function formatCost(amount) {
  return amount < 0.0001 ? `$${amount.toFixed(7)}` : `$${amount.toFixed(5)}`;
}

function formatRate(rate) {
  return `$${Number(rate).toFixed(rate < 1 ? 3 : 2)}`;
}

function updateTokenEstimate() {
  elements.tokenEstimate.textContent = `Estimated input: ${estimateTokens(elements.query.value).toLocaleString()} tokens`;
}

elements.query.addEventListener("input", updateTokenEstimate);

function showError(message) {
  elements.error.textContent = message;
}

function renderResults(data, useCaseLabel) {
  const recommended = data.recommendation;
  elements.task.textContent = useCaseLabel;
  elements.recommendation.replaceChildren();

  if (!recommended) {
    const title = document.createElement("h3");
    title.textContent = "No matching model found";
    const details = document.createElement("p");
    details.className = "recommendation-reason";
    details.textContent = "Add a model for this use case in data/models.json.";
    elements.recommendation.append(title, details);
    elements.modelList.replaceChildren();
    elements.results.hidden = false;
    return;
  }

  const title = document.createElement("h3");
  title.textContent = recommended.name;
  const provider = document.createElement("div");
  provider.className = "provider";
  provider.textContent = `${recommended.provider} · lowest estimated cost for ${useCaseLabel.toLowerCase()}`;
  const cost = document.createElement("div");
  cost.className = "cost";
  cost.textContent = `${formatCost(recommended.estimatedCost)} estimated`;
  const usage = document.createElement("div");
  usage.className = "cost-caption";
  usage.textContent = `About ${data.inputTokens.toLocaleString()} input tokens + ${data.outputTokens.toLocaleString()} output tokens`;
  const rates = document.createElement("p");
  rates.className = "recommendation-reason";
  rates.textContent = `Sample rates: ${formatRate(recommended.inputPerMillion)} per 1M input tokens and ${formatRate(recommended.outputPerMillion)} per 1M output tokens.`;
  elements.recommendation.append(title, provider, cost, usage, rates);

  elements.modelList.replaceChildren();
  data.models.forEach(model => {
    const row = document.createElement("div");
    row.className = "model-row";

    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "model-title";
    name.append(document.createTextNode(model.name));
    if (model.id === recommended.id) {
      const badge = document.createElement("span");
      badge.className = "model-tag";
      badge.textContent = "RECOMMENDED";
      name.append(badge);
    }
    const meta = document.createElement("div");
    meta.className = "model-meta";
    meta.textContent = `${model.provider} · ${formatRate(model.inputPerMillion)} / 1M input · ${formatRate(model.outputPerMillion)} / 1M output`;
    info.append(name, meta);

    const price = document.createElement("div");
    price.className = "model-price";
    price.append(document.createTextNode(formatCost(model.estimatedCost)));
    const caption = document.createElement("small");
    caption.textContent = "for this prompt";
    price.append(caption);

    row.append(info, price);
    elements.modelList.append(row);
  });

  elements.results.hidden = false;
}

async function loadUseCases() {
  const response = await fetch("/api/models", { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error("Could not load the use-case list.");
  const catalog = await response.json();

  elements.useCase.replaceChildren();
  catalog.useCases.forEach(useCase => {
    const option = document.createElement("option");
    option.value = useCase.id;
    option.textContent = useCase.label;
    elements.useCase.append(option);
  });
  elements.useCase.value = catalog.useCases.some(item => item.id === "general") ? "general" : catalog.useCases[0]?.id;
  elements.useCase.disabled = false;
  elements.button.disabled = false;
}

elements.button.addEventListener("click", async () => {
  const prompt = elements.query.value.trim();
  if (!prompt) {
    showError("Enter a prompt or task description first.");
    elements.query.focus();
    elements.results.hidden = true;
    return;
  }

  showError("");
  elements.button.disabled = true;
  elements.button.querySelector("span").textContent = "Calculating recommendation…";

  try {
    // Only token counts and the chosen category are sent. The prompt itself stays in-browser.
    const response = await fetch("/api/quote", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        inputTokens: estimateTokens(prompt),
        outputTokens: Number(elements.outputTokens.value),
        useCase: elements.useCase.value
      })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not calculate a recommendation.");

    const selected = elements.useCase.selectedOptions[0];
    renderResults(result, selected ? selected.textContent : result.useCase);
    elements.results.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error.message || "Something went wrong. Please try again.");
  } finally {
    elements.button.disabled = false;
    elements.button.querySelector("span").textContent = "Find my best-fit model";
  }
});

loadUseCases().catch(error => {
  showError(`${error.message} Start the app with npm start, then open http://localhost:3000.`);
});

updateTokenEstimate();
