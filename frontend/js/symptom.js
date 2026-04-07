(function () {
    const input         = document.getElementById("symptom-input");
    const analyzeBtn    = document.getElementById("symptom-analyze-btn");
    const loader        = document.getElementById("symptom-loader");
    const resultContent = document.getElementById("symptom-result-content");
    const placeholder   = document.getElementById("symptom-result-placeholder");
    const matchedEl     = document.getElementById("symptom-matched");
    const predictionsEl = document.getElementById("symptom-predictions");
    const shapChart     = document.getElementById("symptom-shap-chart");
    const resultPanel   = document.querySelector("#tab-symptom .result-panel");

    let errorEl   = null;
    let summaryEl = null;

    analyzeBtn.addEventListener("click", () => runAnalysis());

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && e.ctrlKey) runAnalysis();
    });

    async function runAnalysis() {
        const text = input.value.trim();
        if (!text) return;

        showLoader();
        analyzeBtn.disabled = true;

        try {
            const result = await apiPost("/api/symptom/predict", { text });

            if (result.error) {
                hideLoader();
                showError(result.error);
            } else {
                hideLoader();
                renderResult(result);
            }
        } catch (err) {
            hideLoader();
            showError(err.message);
        } finally {
            analyzeBtn.disabled = false;
        }
    }

    function renderResult(data) {
        hideError();
            if (data.low_confidence) {
        let warnEl = document.createElement("div");
        warnEl.className = "warn-box";
        warnEl.innerHTML = `
            <i class="ph ph-warning"></i>
            <span>Nəticə qeyri-müəyyəndir. Daha dəqiq diaqnoz üçün əlavə simptomlar yazın.</span>
        `;
        resultContent.insertBefore(warnEl, resultContent.firstChild);
    }

        matchedEl.innerHTML = data.matched_symptoms
            .map((s, i) => `
                <span class="tag" style="animation-delay:${i * 0.05}s">
                    ${s.replace(/_/g, " ")}
                </span>
            `)
            .join("");

        predictionsEl.innerHTML = data.predictions
            .map((p, i) => `
                <div class="prediction-item" style="animation-delay:${i * 0.1}s">
                    <span class="prediction-rank">${i + 1}</span>
                    <span class="prediction-name">${p.disease}</span>
                    <div class="prediction-bar-wrap">
                        <div class="prediction-bar" style="width:${(p.probability * 100).toFixed(1)}%"></div>
                    </div>
                    <span class="prediction-pct">${(p.probability * 100).toFixed(1)}%</span>
                </div>
            `)
            .join("");

        if (data.shap_chart_b64) {
            shapChart.src = b64ToImgSrc(data.shap_chart_b64);
        }

        if (data.llm_summary) {
            if (!summaryEl) {
                summaryEl           = document.createElement("div");
                summaryEl.id        = "symptom-llm-summary";
                summaryEl.className = "llm-summary-box";
                resultContent.appendChild(summaryEl);
            }
            summaryEl.innerHTML = `
                <div class="box-title">
                    <i class="ph ph-sparkle"></i>
                    <h4>AI İzahı</h4>
                </div>
                <p>${data.llm_summary.replace(/\n/g, "<br>")}</p>
            `;
            summaryEl.classList.remove("hidden");
        }

        const top = data.predictions[0];
        setDiagnosisContext(
            `[Symptom Check] Top diagnosis: ${top.disease} (${(top.probability * 100).toFixed(1)}%) | Symptoms: ${data.matched_symptoms.join(", ")}`
        );

        placeholder.classList.add("hidden");
        resultContent.classList.remove("hidden");
    }

    function showLoader() {
        resultContent.classList.add("hidden");
        placeholder.classList.add("hidden");
        hideError();
        loader.classList.remove("hidden");
    }

    function hideLoader() {
        loader.classList.add("hidden");
    }

    function showError(msg) {
        resultContent.classList.add("hidden");
        placeholder.classList.add("hidden");

        if (!errorEl) {
            errorEl           = document.createElement("div");
            errorEl.className = "error-box";
            resultPanel.appendChild(errorEl);
        }

        errorEl.innerHTML = `
            <i class="ph ph-warning-circle"></i>
            <div class="error-content">
                <span class="error-title">Simptom tanınmadı</span>
                <span class="error-msg">${msg}</span>
            </div>
        `;
        errorEl.classList.remove("hidden");
    }

    function hideError() {
        if (errorEl) errorEl.classList.add("hidden");
    }
})();