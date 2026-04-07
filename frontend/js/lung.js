(function () {
    const uploadArea    = document.getElementById("lung-upload-area");
    const fileInput     = document.getElementById("lung-file");
    const placeholder   = document.getElementById("lung-placeholder");
    const preview       = document.getElementById("lung-preview");
    const analyzeBtn    = document.getElementById("lung-analyze-btn");
    const loader        = document.getElementById("lung-loader");
    const resultContent = document.getElementById("lung-result-content");
    const resultPanel   = document.querySelector("#tab-lung .result-panel");
    const badge         = document.getElementById("lung-badge");
    const probChart     = document.getElementById("lung-prob-chart");
    const heatmap       = document.getElementById("lung-heatmap");

    let selectedFile = null;
    let errorEl      = null;

    uploadArea.addEventListener("click", () => fileInput.click());

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("image/")) handleFile(file);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        selectedFile = file;
        preview.src  = URL.createObjectURL(file);
        preview.classList.remove("hidden");
        placeholder.classList.add("hidden");
        analyzeBtn.disabled = false;
        hideError();
    }

    analyzeBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        showLoader();
        analyzeBtn.disabled = true;

        try {
            const b64    = await imageFileToB64(selectedFile);
            const result = await apiPost("/api/lung/predict", { image_b64: b64 });
            renderResult(result);
        } catch (err) {
            showError(err.message);
        } finally {
            hideLoader();
            analyzeBtn.disabled = false;
        }
    });

    function renderResult(data) {
        const isPneumonia = data.label === "PNEUMONIA";

        badge.className = "diagnosis-badge " + (isPneumonia ? "danger" : "normal");
        badge.innerHTML = `
            <i class="ph ph-${isPneumonia ? "warning" : "check-circle"}"></i>
            <span>${isPneumonia ? "Pnevmoniya Aşkarlandı" : "Normal"}</span>
            <span style="margin-left:auto;font-size:14px;opacity:0.8">
                ${(data.confidence * 100).toFixed(1)}%
            </span>
        `;

        probChart.src = b64ToImgSrc(data.prob_chart_b64);
        heatmap.src   = b64ToImgSrc(data.heatmap_b64);

        setDiagnosisContext(
            `[Lung X-Ray] Prediction: ${data.label} | Confidence: ${(data.confidence * 100).toFixed(1)}%`
        );

        hideError();
        resultContent.classList.remove("hidden");
    }

    function showLoader() {
        resultContent.classList.add("hidden");
        hideError();
        loader.classList.remove("hidden");
    }

    function hideLoader() {
        loader.classList.add("hidden");
    }

    function showError(msg) {
        resultContent.classList.add("hidden");

        if (!errorEl) {
            errorEl           = document.createElement("div");
            errorEl.className = "error-box";
            resultPanel.appendChild(errorEl);
        }

        errorEl.innerHTML = `
            <i class="ph ph-warning-circle"></i>
            <span>${msg}</span>
        `;
        errorEl.classList.remove("hidden");
    }

    function hideError() {
        if (errorEl) errorEl.classList.add("hidden");
    }
})();