(function () {
    const uploadArea    = document.getElementById("brain-upload-area");
    const fileInput     = document.getElementById("brain-file");
    const placeholder   = document.getElementById("brain-placeholder");
    const preview       = document.getElementById("brain-preview");
    const analyzeBtn    = document.getElementById("brain-analyze-btn");
    const loader        = document.getElementById("brain-loader");
    const resultContent = document.getElementById("brain-result-content");
    const resultPanel   = document.querySelector("#tab-brain .result-panel");
    const badge         = document.getElementById("brain-badge");
    const probChart     = document.getElementById("brain-prob-chart");
    const gradcam       = document.getElementById("brain-gradcam");

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
            const result = await apiPost("/api/brain/predict", { image_b64: b64 });
            renderResult(result);
        } catch (err) {
            showError(err.message);
        } finally {
            hideLoader();
            analyzeBtn.disabled = false;
        }
    });

    function renderResult(data) {
        const colorMap = {
            "No Tumor":        "normal",
            "Glioma":          "danger",
            "Meningioma":      "warning",
            "Pituitary Tumor": "purple",
        };
        const iconMap = {
            "No Tumor":        "check-circle",
            "Glioma":          "warning",
            "Meningioma":      "warning-circle",
            "Pituitary Tumor": "info",
        };

        badge.className = "diagnosis-badge " + (colorMap[data.label] || "normal");
        badge.innerHTML = `
            <i class="ph ph-${iconMap[data.label] || "info"}"></i>
            <span>${data.label}</span>
            <span style="margin-left:auto;font-size:14px;opacity:0.8">
                ${(data.confidence * 100).toFixed(1)}%
            </span>
        `;

        probChart.src = b64ToImgSrc(data.prob_chart_b64);
        gradcam.src   = b64ToImgSrc(data.gradcam_b64);

        setDiagnosisContext(
            `[Brain MRI] Prediction: ${data.label} | Confidence: ${(data.confidence * 100).toFixed(1)}%`
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