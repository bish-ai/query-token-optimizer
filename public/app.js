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
    const spheres = [
      { position: [-2.7, 0.8, 0], radius: 1.05, color: "#9cb0ff" },
      { position: [2.4, -0.5, -0.4], radius: 0.85, color: "#79e0c0" },
      { position: [0.2, 1.6, -0.8], radius: 1.25, color: "#b3c1ff" }
    ].map(({ position, radius, color }) => {
      const mesh = new Mesh(
        new SphereGeometry(radius, 24, 24),
        new MeshBasicMaterial({ color: new Color(color), wireframe: true, transparent: true, opacity: 0.42 })
      );
      mesh.position.set(...position);
      group.add(mesh);
      return mesh;
    });
    const resize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight, false);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener("resize", resize);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame = 0;
    const animate = () => {
      frame += reduceMotion ? 0 : 0.003;
      spheres.forEach((mesh, i) => {
        mesh.rotation.x += reduceMotion ? 0 : 0.0007 * (i + 1);
        mesh.rotation.y += reduceMotion ? 0 : 0.001 * (i + 1);
      });
      group.rotation.z = Math.sin(frame) * 0.08;
      renderer.render(scene, camera);
      if (!reduceMotion) requestAnimationFrame(animate);
    };
    animate();
  })
  .catch(() => {
    // The app remains usable when WebGL or the Three.js CDN is unavailable.
  });

/*
 * Lightweight, in-browser supervised text classifier.
 * This is a multinomial Naive Bayes model trained on the small example corpus below.
 * It does not send the prompt anywhere and is intended as a useful starting heuristic,
 * not a substitute for a large, evaluated production classifier.
 */
const TRAINING_EXAMPLES = {
  simple: [
    "extract names dates amounts and key fields from text",
    "classify these messages into categories or labels",
    "rewrite this sentence fix grammar and spelling",
    "convert this list into a structured format",
    "translate this phrase or short passage",
    "identify sentiment positive negative or neutral",
    "remove duplicates and normalize these records",
    "answer a simple factual question from the provided text"
  ],
  general: [
    "answer a general question and explain a topic clearly",
    "summarize this article document or long report",
    "help me understand this subject and give an overview",
    "compare these options and provide useful information",
    "give me recommendations for everyday planning",
    "explain the main points in simple language",
    "respond to a question using the context provided",
    "provide a helpful overview of this topic"
  ],
  writing: [
    "write a professional email to a customer",
    "draft a blog post story essay or marketing copy",
    "edit this paragraph for clarity and style",
    "make this message more persuasive and engaging",
    "change the tone to be friendly concise or formal",
    "brainstorm creative ideas and headlines",
    "write a product description or social media post",
    "proofread and improve this draft"
  ],
  coding: [
    "write a javascript python or typescript function",
    "debug this code and fix the error or exception",
    "explain this program and how the algorithm works",
    "refactor this function and improve its performance",
    "create a sql query or api endpoint",
    "review this code and identify a software bug",
    "write unit tests for this module",
    "help me build a web application component"
  ],
  reasoning: [
    "solve this difficult multi step logic problem",
    "analyze the evidence and evaluate competing hypotheses",
    "derive a rigorous proof or mathematical solution",
    "plan a complex project with constraints and dependencies",
    "reason through a difficult strategic decision",
    "perform a detailed quantitative analysis",
    "find edge cases in this complicated system design",
    "compare scenarios and explain the tradeoffs step by step"
  ]
};

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "give",
  "help", "i", "in", "into", "is", "it", "me", "of", "on", "or", "please",
  "this", "that", "the", "these", "to", "using", "with", "you", "your"
]);

function tokenize(text) {
  return (text.toLowerCase().match(/[a-z0-9+#_.-]+/g) || [])
    .filter(token => token.length > 1 && !STOP_WORDS.has(token));
}

function trainNaiveBayes(examplesByClass) {
  const classes = {};
  const vocabulary = new Set();

  for (const [label, examples] of Object.entries(examplesByClass)) {
    const wordCounts = new Map();
    let totalWords = 0;
    for (const example of examples) {
      for (const word of tokenize(example)) {
        vocabulary.add(word);
        wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
        totalWords++;
      }
    }
    classes[label] = { exampleCount: examples.length, wordCounts, totalWords };
  }
  return { classes, vocabulary, totalExamples: Object.values(classes).reduce((sum, item) => sum + item.exampleCount, 0) };
}

const classifier = trainNaiveBayes(TRAINING_EXAMPLES);

function classifyUseCase(text) {
  const words = tokenize(text);
  const classEntries = Object.entries(classifier.classes);
  const scored = classEntries.map(([label, stats]) => {
    // Laplace-smoothed multinomial Naive Bayes with a uniform feature-independent prior.
    let logScore = Math.log(stats.exampleCount / classifier.totalExamples);
    const denominator = stats.totalWords + classifier.vocabulary.size;
    for (const word of words) {
      logScore += Math.log(((stats.wordCounts.get(word) || 0) + 1) / denominator);
    }
    return { label, logScore };
  });

  // Softmax gives a useful relative confidence estimate from the log scores.
  const maxScore = Math.max(...scored.map(item => item.logScore));
  const probabilities = scored.map(item => ({
    ...item,
    probability: Math.exp(item.logScore - maxScore)
  }));
  const normalizer = probabilities.reduce((sum, item) => sum + item.probability, 0);
  probabilities.forEach(item => { item.probability /= normalizer; });
  probabilities.sort((a, b) => b.probability - a.probability);

  return {
    useCase: probabilities[0].label,
    confidence: probabilities[0].probability,
    alternatives: probabilities.slice(1, 3).map(item => ({
      useCase: item.label,
      confidence: item.probability
    }))
  };
}

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

function renderResults(data, useCaseLabel, detection) {
  const recommended = data.recommendation;
  elements.task.textContent = detection
    ? `${useCaseLabel} · ML ${Math.round(detection.confidence * 100)}%`
    : useCaseLabel;
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
  rates.textContent = detection
    ? `Use case automatically detected by the in-browser Naive Bayes classifier (${Math.round(detection.confidence * 100)}% relative confidence). Sample rates: ${formatRate(recommended.inputPerMillion)} per 1M input and ${formatRate(recommended.outputPerMillion)} per 1M output tokens.`
    : `Use case selected manually. Sample rates: ${formatRate(recommended.inputPerMillion)} per 1M input and ${formatRate(recommended.outputPerMillion)} per 1M output tokens.`;
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
  const autoOption = document.createElement("option");
  autoOption.value = "auto";
  autoOption.textContent = "Auto-detect with ML";
  elements.useCase.append(autoOption);
  catalog.useCases.forEach(useCase => {
    const option = document.createElement("option");
    option.value = useCase.id;
    option.textContent = useCase.label;
    elements.useCase.append(option);
  });
  elements.useCase.value = "auto";
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
  elements.button.querySelector("span").textContent = "Detecting use case and comparing models…";

  try {
    const selectedUseCase = elements.useCase.value;
    const detection = selectedUseCase === "auto" ? classifyUseCase(prompt) : null;
    const useCase = detection ? detection.useCase : selectedUseCase;
    const response = await fetch("/api/quote", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      // Only token counts/category are sent; prompt content stays in the browser.
      body: JSON.stringify({
        inputTokens: estimateTokens(prompt),
        outputTokens: Number(elements.outputTokens.value),
        useCase
      })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not calculate a recommendation.");

    const useCaseOption = [...elements.useCase.options].find(option => option.value === useCase);
    const useCaseLabel = useCaseOption?.textContent || useCase;
    renderResults(result, useCaseLabel, detection);
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
