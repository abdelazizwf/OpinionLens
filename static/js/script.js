/**
 * OpinionLens - Sentiment Analysis Frontend
 */

document.addEventListener("DOMContentLoaded", () => {
    const submitBtn = document.getElementById("submitBtn");
    const inputText = document.getElementById("inputText");
    const resultDiv = document.getElementById("result");
    const errorDiv = document.getElementById("error");
    const latencyDiv = document.getElementById("latency");
    const themeToggle = document.getElementById("themeToggle");
    const root = document.documentElement;

    /**
     * Theme Initialization and Handling
     */
    const initTheme = () => {
        const savedTheme = localStorage.getItem("theme") || "dark";
        root.setAttribute("data-theme", savedTheme);
        themeToggle.textContent = savedTheme === "dark" ? "🌙" : "☀️";
    };

    const toggleTheme = () => {
        const currentTheme = root.getAttribute("data-theme");
        const nextTheme = currentTheme === "dark" ? "light" : "dark";

        root.setAttribute("data-theme", nextTheme);
        localStorage.setItem("theme", nextTheme);
        themeToggle.textContent = nextTheme === "dark" ? "🌙" : "☀️";
    };

    /**
     * Inference Logic
     */
    const evaluateSentiment = async () => {
        const text = inputText.value.trim();

        // Clear previous state
        errorDiv.textContent = "";
        resultDiv.style.display = "none";
        latencyDiv.style.display = "none";

        if (!text) {
            errorDiv.textContent = "Please enter some text before submitting.";
            return;
        }

        // UI Feedback - Start
        submitBtn.disabled = true;
        submitBtn.textContent = "Evaluating...";

        try {
            const startTime = performance.now();

            const response = await fetch("/api/v1/inference/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error(`Inference request failed with status: ${response.status}`);
            }

            const endTime = performance.now();
            const latencyMs = Math.round(endTime - startTime);
            const data = await response.json();
            const prediction = data.prediction;

            // Render Result
            resultDiv.textContent = prediction;
            resultDiv.className = "result"; // Reset classes
            resultDiv.classList.add(prediction === "POSITIVE" ? "positive" : "negative");
            resultDiv.style.display = "block";

            // Render Latency
            latencyDiv.textContent = `Latency: ${latencyMs} ms`;
            latencyDiv.style.display = "inline";

        } catch (err) {
            console.error("Inference error:", err);
            errorDiv.textContent = "Failed to evaluate text. Check the API or network.";
        } finally {
            // UI Feedback - End
            submitBtn.disabled = false;
            submitBtn.textContent = "Evaluate";
        }
    };

    /**
     * Event Listeners
     */
    themeToggle.addEventListener("click", toggleTheme);

    submitBtn.addEventListener("click", evaluateSentiment);

    inputText.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") {
            e.preventDefault();
            evaluateSentiment();
        }
    });

    // Initialize UI
    initTheme();
});
