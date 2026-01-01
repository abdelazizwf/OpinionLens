const submitBtn = document.getElementById("submitBtn");
const inputText = document.getElementById("inputText");
const resultDiv = document.getElementById("result");
const errorDiv = document.getElementById("error");
const themeToggle = document.getElementById("themeToggle");
const root = document.documentElement;

const savedTheme = localStorage.getItem("theme") || "dark";
root.setAttribute("data-theme", savedTheme);
themeToggle.textContent = savedTheme === "dark" ? "🌙" : "☀️";

themeToggle.addEventListener("click", () => {
    const current = root.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";

    root.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    themeToggle.textContent = next === "dark" ? "🌙" : "☀️";
});

inputText.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
    e.preventDefault();
    submitBtn.click();
    }
});

submitBtn.addEventListener("click", async () => {
    const text = inputText.value.trim();

    errorDiv.textContent = "";
    resultDiv.style.display = "none";

    if (!text) {
    errorDiv.textContent = "Please enter some text before submitting.";
    return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = "Evaluating...";

    try {
    const latencyDiv = document.getElementById("latency");

    const startTime = performance.now();

    const response = await fetch("/api/v1/inference/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    const endTime = performance.now();
    const latencyMs = Math.round(endTime - startTime);

    const data = await response.json();
    const prediction = data.prediction;

    resultDiv.className = "result";
    resultDiv.textContent = prediction;

    if (prediction === "POSITIVE") {
        resultDiv.classList.add("positive");
    } else {
        resultDiv.classList.add("negative");
    }

    resultDiv.style.display = "block";
    latencyDiv.textContent = `Latency: ${latencyMs} ms`;
    latencyDiv.style.display = "block";
    } catch (err) {
    errorDiv.textContent = "Failed to evaluate text. Check the API or network.";
    console.error(err);
    } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Evaluate";
    }
});
