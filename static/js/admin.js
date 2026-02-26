/**
 * OpinionLens - Admin Panel Frontend
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const themeToggle = document.getElementById("themeToggle");
    const root = document.documentElement;
    const modelsTableBody = document.querySelector("#modelsTable tbody");
    const loadedModelsTableBody = document.querySelector("#loadedModelsTable tbody");
    const modelNameSelect = document.getElementById("modelNameSelect");
    const modelVersionSelect = document.getElementById("modelVersionSelect");
    const downloadForm = document.getElementById("downloadForm");
    const downloadBtn = document.getElementById("downloadBtn");
    const downloadSpinner = document.getElementById("downloadSpinner");
    const setDefaultCheckbox = document.getElementById("setDefaultCheckbox");

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
     * Data Loading and Rendering
     */
    const fetchAvailableModels = async () => {
        try {
            const response = await fetch("/api/v1/models/registry");
            if (!response.ok) throw new Error(`Status: ${response.status}`);

            const models = await response.json();
            localStorage.setItem("availableModels", JSON.stringify(models));
            renderRegistryTable(models);
            populateModelNames(models);
        } catch (err) {
            console.error("Failed to load available models:", err);
        }
    };

    const fetchLoadedModels = async () => {
        try {
            const response = await fetch("/api/v1/models/?brief=true");
            if (!response.ok) throw new Error(`Status: ${response.status}`);

            const models = await response.json();
            localStorage.setItem("loadedModels", JSON.stringify(models));
            renderLoadedTable(models);
        } catch (err) {
            console.error("Failed to load loaded models:", err);
        }
    };

    const renderRegistryTable = (models) => {
        modelsTableBody.innerHTML = "";
        models.forEach(model => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${model.name}</td>
                <td>${model.latest_version}</td>
                <td>${new Date(model.latest_version_creation).toLocaleString()}</td>
            `;
            modelsTableBody.appendChild(row);
        });
    };

    const renderLoadedTable = (models) => {
        loadedModelsTableBody.innerHTML = "";
        models.forEach(model => {
            const row = document.createElement("tr");
            if (model.is_default) row.classList.add("active-model");

            row.innerHTML = `
                <td>${model.name}</td>
                <td>${model.version}</td>
                <td>${new Date(model.creation).toLocaleString()}</td>
                <td>
                    <span class="status-badge ${model.is_default ? "status-active" : "status-inactive"}">
                        ${model.is_default ? "Active" : "Inactive"}
                    </span>
                </td>
                <td>
                    ${model.is_default ? "" : `
                        <button class="set-active-btn" data-name="${model.name}" data-version="${model.version}" title="Set active">⭐</button>
                        <button class="delete-btn" data-id="${model.model_id}" title="Delete model">🗑️</button>
                    `}
                </td>
            `;
            loadedModelsTableBody.appendChild(row);
        });
    };

    /**
     * Form and Actions
     */
    const populateModelNames = (availableModels) => {
        const loadedModels = JSON.parse(localStorage.getItem("loadedModels")) || [];
        const loadedByName = loadedModels.reduce((acc, model) => {
            if (!acc[model.name]) acc[model.name] = new Set();
            acc[model.name].add(model.version);
            return acc;
        }, {});

        const uniqueNames = [...new Set(availableModels.map(m => m.name))];
        modelNameSelect.innerHTML = `<option value="">Select a model</option>`;

        uniqueNames.forEach(name => {
            const registryModel = availableModels.find(m => m.name === name);
            const loadedVersions = loadedByName[name] || new Set();

            if (registryModel.latest_version === 1 && loadedVersions.has(1)) return;

            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            modelNameSelect.appendChild(option);
        });
    };

    const updateSubmitState = () => {
        const hasName = Boolean(modelNameSelect.value);
        const hasVersion = !modelVersionSelect.disabled && Boolean(modelVersionSelect.value);
        downloadBtn.disabled = !(hasName && hasVersion);
    };

    /**
     * Event Listeners
     */
    themeToggle.addEventListener("click", toggleTheme);

    modelNameSelect.addEventListener("change", () => {
        const selectedName = modelNameSelect.value;
        const availableModels = JSON.parse(localStorage.getItem("availableModels")) || [];
        const loadedModels = JSON.parse(localStorage.getItem("loadedModels")) || [];
        const loadedVersions = new Set(loadedModels.filter(m => m.name === selectedName).map(m => m.version));

        modelVersionSelect.innerHTML = `<option value="">Select version</option>`;
        modelVersionSelect.disabled = true;

        if (!selectedName) return;

        const registryModel = availableModels.find(m => m.name === selectedName);
        if (registryModel) {
            for (let v = 1; v <= registryModel.latest_version; v++) {
                if (loadedVersions.has(v)) continue;
                const option = document.createElement("option");
                option.value = v;
                option.textContent = v;
                modelVersionSelect.appendChild(option);
            }
        }

        if (modelVersionSelect.options.length > 1) {
            modelVersionSelect.disabled = false;
        }
        updateSubmitState();
    });

    modelVersionSelect.addEventListener("change", updateSubmitState);

    downloadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = modelNameSelect.value;
        const version = Number(modelVersionSelect.value);
        const setDefault = setDefaultCheckbox.checked;

        if (!name || !version) return;

        downloadBtn.disabled = true;
        downloadSpinner.style.display = "block";

        try {
            const response = await fetch("/api/v1/models/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_name: name, model_version: version, set_default: setDefault })
            });

            if (response.status === 201) {
                window.location.reload();
            } else {
                console.error("Unexpected response:", response.status);
            }
        } catch (err) {
            console.error("Failed to download model:", err);
        } finally {
            downloadSpinner.style.display = "none";
        }
    });

    loadedModelsTableBody.addEventListener("click", async (e) => {
        const setActiveBtn = e.target.closest(".set-active-btn");
        const deleteBtn = e.target.closest(".delete-btn");

        if (setActiveBtn) {
            const name = setActiveBtn.dataset.name;
            const version = Number(setActiveBtn.dataset.version);
            try {
                const response = await fetch("/api/v1/models/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model_name: name, model_version: version, set_default: true })
                });
                if (response.status === 201) window.location.reload();
            } catch (err) {
                console.error("Set active failed:", err);
            }
        }

        if (deleteBtn) {
            const modelId = deleteBtn.dataset.id;
            try {
                const response = await fetch(`/api/v1/models/?model_id=${modelId}`, { method: "DELETE" });
                if (response.status === 200) window.location.reload();
            } catch (err) {
                console.error("Delete failed:", err);
            }
        }
    });

    // Initialize
    initTheme();
    fetchLoadedModels().then(fetchAvailableModels);
});
