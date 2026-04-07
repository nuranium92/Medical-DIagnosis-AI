const BASE_URL = "http://localhost:8000";

async function apiPost(endpoint, body) {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${response.status}`);
    }

    return response.json();
}

function imageFileToB64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload  = () => {
            const b64 = reader.result.split(",")[1];
            resolve(b64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function b64ToImgSrc(b64) {
    return `data:image/png;base64,${b64}`;
}

let lastDiagnosisContext = "";

function setDiagnosisContext(ctx) {
    lastDiagnosisContext = ctx || "";
}

function getDiagnosisContext() {
    return lastDiagnosisContext;
}