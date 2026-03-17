/**
 * OpinionLens - Sentiment Analysis Frontend
 */

document.addEventListener("DOMContentLoaded", () => {
    // Single Input Elements
    const submitBtn = document.getElementById("submitBtn");
    const inputText = document.getElementById("inputText");
    const resultDiv = document.getElementById("result");
    const latencyDiv = document.getElementById("latency");

    // Batch Input Elements
    const uploadBtn = document.getElementById("uploadBtn");
    const fileInput = document.getElementById("fileInput");
    const delimiterSelect = document.getElementById("delimiter");
    const batchResultDiv = document.getElementById("batchResult");
    const resultsTableBody = document.querySelector("#resultsTable tbody");
    const positiveCountSpan = document.getElementById("positiveCount");
    const negativeCountSpan = document.getElementById("negativeCount");
    const batchLatencyDiv = document.getElementById("batchLatency");

    // CSV Input Elements
    const csvUploadBtn = document.getElementById("csvUploadBtn");
    const csvDownloadBtn = document.getElementById("csvDownloadBtn");
    const csvFileInput = document.getElementById("csvFileInput");
    const columnNameInput = document.getElementById("columnName");
    const csvLatencyDiv = document.getElementById("csvLatency");

    // Pagination Elements
    const prevPageBtn = document.getElementById("prevPage");
    const nextPageBtn = document.getElementById("nextPage");
    const pageInfoSpan = document.getElementById("pageInfo");

    // Shared Elements
    const errorDiv = document.getElementById("error");
    const themeToggle = document.getElementById("themeToggle");
    const root = document.documentElement;

    // Tab Elements
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    // Pagination State
    let currentBatchData = {
        predictions: [],
        segments: []
    };
    let currentPage = 1;
    const itemsPerPage = 5;

    // CSV Cache
    let cachedCsvBlob = null;
    let cachedCsvFileName = "";

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
     * Tab Switching
     */
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const tabId = btn.getAttribute("data-tab");

            // Update buttons
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Update content
            tabContents.forEach(content => {
                content.classList.remove("active");
                if (content.id === `${tabId}Input`) {
                    content.classList.add("active");
                }
            });

            // Clear results when switching
            clearStates();
        });
    });

    const clearStates = () => {
        errorDiv.textContent = "";
        resultDiv.style.display = "none";
        latencyDiv.style.display = "none";
        batchResultDiv.style.display = "none";
        batchLatencyDiv.style.display = "none";
        csvLatencyDiv.style.display = "none";
        csvDownloadBtn.style.display = "none";
        currentBatchData = { predictions: [], segments: [] };
        currentPage = 1;

        // Clear CSV cache
        if (cachedCsvBlob) {
            window.URL.revokeObjectURL(cachedCsvBlob);
            cachedCsvBlob = null;
            cachedCsvFileName = "";
        }
    };

    /**
     * File Selection Feedback
     */
    fileInput.addEventListener("change", (e) => {
        const fileName = e.target.files[0]?.name || "Choose a text file...";
        document.querySelector("#batchInput .file-label span").textContent = fileName;
    });

    csvFileInput.addEventListener("change", (e) => {
        const fileName = e.target.files[0]?.name || "Choose a CSV file...";
        document.querySelector("#csvInput .file-label span").textContent = fileName;
        // Hide download button when a new file is selected
        csvDownloadBtn.style.display = "none";
    });

    /**
     * Single Inference Logic
     */
    const evaluateSentiment = async () => {
        const text = inputText.value.trim();

        clearStates();

        if (!text) {
            errorDiv.textContent = "Please enter some text before submitting.";
            return;
        }

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
                const data = await response.json();
                throw new Error(data.detail || `Inference request failed with status: ${response.status}`);
            }

            const endTime = performance.now();
            const latencyMs = Math.round(endTime - startTime);
            const data = await response.json();
            const prediction = data.prediction;

            resultDiv.textContent = prediction;
            resultDiv.className = "result";
            resultDiv.classList.add(prediction === "POSITIVE" ? "positive" : "negative");
            resultDiv.style.display = "block";

            latencyDiv.textContent = `Latency: ${latencyMs} ms`;
            latencyDiv.style.display = "inline";

        } catch (err) {
            console.error("Inference error:", err);
            errorDiv.textContent = err.message || "Failed to evaluate text. Check the API or network.";
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Evaluate";
        }
    };

    /**
     * Batch Rendering with Pagination
     */
    const renderBatchPage = (page) => {
        const startIndex = (page - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const pagePredictions = currentBatchData.predictions.slice(startIndex, endIndex);
        const pageSegments = currentBatchData.segments.slice(startIndex, endIndex);

        resultsTableBody.innerHTML = "";
        pagePredictions.forEach((prediction, index) => {
            const row = document.createElement("tr");
            const textCell = document.createElement("td");
            const sentimentCell = document.createElement("td");

            textCell.textContent = pageSegments[index] || "(empty segment)";
            sentimentCell.textContent = prediction;
            sentimentCell.className = prediction === "POSITIVE" ? "positive" : "negative";

            row.appendChild(textCell);
            row.appendChild(sentimentCell);
            resultsTableBody.appendChild(row);
        });

        const totalPages = Math.ceil(currentBatchData.predictions.length / itemsPerPage);
        pageInfoSpan.textContent = `Page ${page} of ${totalPages}`;
        prevPageBtn.disabled = page === 1;
        nextPageBtn.disabled = page === totalPages || totalPages === 0;
    };

    /**
     * Batch Inference Logic
     */
    const uploadAndEvaluate = async () => {
        const file = fileInput.files[0];
        const delimiter = delimiterSelect.value;

        clearStates();

        if (!file) {
            errorDiv.textContent = "Please select a file before uploading.";
            return;
        }

        uploadBtn.disabled = true;
        uploadBtn.textContent = "Uploading & Evaluating...";

        const formData = new FormData();
        formData.append("file", file);
        formData.append("delimiter", delimiter);

        try {
            const startTime = performance.now();

            const response = await fetch("/upload_txt", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || `Upload failed with status: ${response.status}`);
            }

            const endTime = performance.now();
            const latencyMs = Math.round(endTime - startTime);
            const predictions = await response.json();

            // Process results
            const fileContent = await file.text();
            let escapedDelimiter = delimiter;
            if (delimiter === "\\n") escapedDelimiter = "\n";
            else if (delimiter === "\\t") escapedDelimiter = "\t";

            const segments = fileContent.split(escapedDelimiter).map(s => s.trim()).filter(s => s);

            currentBatchData = {
                predictions: predictions,
                segments: segments
            };

            let positiveCount = predictions.filter(p => p === "POSITIVE").length;
            let negativeCount = predictions.length - positiveCount;

            positiveCountSpan.textContent = `Positive: ${positiveCount}`;
            negativeCountSpan.textContent = `Negative: ${negativeCount}`;

            batchResultDiv.style.display = "block";
            batchLatencyDiv.textContent = `Batch Latency: ${latencyMs} ms`;
            batchLatencyDiv.style.display = "inline";

            currentPage = 1;
            renderBatchPage(currentPage);

        } catch (err) {
            console.error("Batch inference error:", err);
            errorDiv.textContent = err.message || "Failed to process batch upload.";
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = "Upload & Evaluate";
        }
    };

    /**
     * CSV Inference Logic
     */
    const uploadCsvAndEvaluate = async () => {
        const file = csvFileInput.files[0];
        const columnName = columnNameInput.value.trim();

        // Clear only non-CSV results but reset CSV state for a new file
        errorDiv.textContent = "";
        resultDiv.style.display = "none";
        batchResultDiv.style.display = "none";
        csvDownloadBtn.style.display = "none";
        csvLatencyDiv.style.display = "none";

        if (!file) {
            errorDiv.textContent = "Please select a CSV file before uploading.";
            return;
        }

        if (!columnName) {
            errorDiv.textContent = "Please enter the name of the text column.";
            return;
        }

        csvUploadBtn.disabled = true;
        csvUploadBtn.textContent = "Processing CSV...";

        const formData = new FormData();
        formData.append("file", file);
        formData.append("column_name", columnName);

        try {
            const startTime = performance.now();

            const response = await fetch("/upload_csv", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || `CSV processing failed with status: ${response.status}`);
            }

            const endTime = performance.now();
            const latencyMs = Math.round(endTime - startTime);

            const blob = await response.blob();

            // Cache the result
            if (cachedCsvBlob) {
                window.URL.revokeObjectURL(cachedCsvBlob);
            }
            cachedCsvBlob = window.URL.createObjectURL(blob);
            cachedCsvFileName = `evaluated_${file.name}`;

            csvDownloadBtn.style.display = "inline-block";
            csvLatencyDiv.textContent = `Processing Time: ${latencyMs} ms`;
            csvLatencyDiv.style.display = "inline";

        } catch (err) {
            console.error("CSV inference error:", err);
            errorDiv.textContent = err.message || "Failed to process CSV file.";
        } finally {
            csvUploadBtn.disabled = false;
            csvUploadBtn.textContent = "Process CSV";
        }
    };

    const downloadCachedCsv = () => {
        if (!cachedCsvBlob) return;

        const a = document.createElement("a");
        a.style.display = "none";
        a.href = cachedCsvBlob;
        a.download = cachedCsvFileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    /**
     * Event Listeners
     */
    themeToggle.addEventListener("click", toggleTheme);
    submitBtn.addEventListener("click", evaluateSentiment);
    uploadBtn.addEventListener("click", uploadAndEvaluate);
    csvUploadBtn.addEventListener("click", uploadCsvAndEvaluate);
    csvDownloadBtn.addEventListener("click", downloadCachedCsv);

    prevPageBtn.addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            renderBatchPage(currentPage);
        }
    });

    nextPageBtn.addEventListener("click", () => {
        const totalPages = Math.ceil(currentBatchData.predictions.length / itemsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            renderBatchPage(currentPage);
        }
    });

    inputText.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "Enter") {
            e.preventDefault();
            evaluateSentiment();
        }
    });

    // Initialize UI
    initTheme();
});
