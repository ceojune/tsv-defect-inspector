// ===== DOM 요소 =====
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const previewSection = document.getElementById("previewSection");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const loadingSection = document.getElementById("loadingSection");
const resultSection = document.getElementById("resultSection");
const summaryCards = document.getElementById("summaryCards");
const originalResult = document.getElementById("originalResult");
const analyzedResult = document.getElementById("analyzedResult");
const defectTable = document.getElementById("defectTable").querySelector("tbody");
const noDefects = document.getElementById("noDefects");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");

let selectedFile = null;

// ===== 업로드 영역 이벤트 =====
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
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

// ===== 파일 처리 =====
function handleFile(file) {
    const validTypes = ["image/png", "image/jpeg", "image/tiff", "image/bmp"];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(tif|tiff)$/i)) {
        alert("지원하지 않는 파일 형식입니다.\nPNG, JPG, TIFF, BMP 파일을 선택해주세요.");
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        alert("파일 크기가 16MB를 초과합니다.");
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = "none";
        previewSection.style.display = "block";
        resultSection.style.display = "none";
    };
    reader.readAsDataURL(file);
}

// ===== 분석 시작 =====
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    previewSection.style.display = "none";
    loadingSection.style.display = "block";
    resultSection.style.display = "none";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (data.error) {
            alert("분석 오류: " + data.error);
            loadingSection.style.display = "none";
            previewSection.style.display = "block";
            analyzeBtn.disabled = false;
            return;
        }

        displayResults(data);
    } catch (err) {
        alert("서버 연결에 실패했습니다. 서버가 실행 중인지 확인해주세요.");
        loadingSection.style.display = "none";
        previewSection.style.display = "block";
    }

    analyzeBtn.disabled = false;
});

// ===== 결과 표시 =====
function displayResults(data) {
    loadingSection.style.display = "none";
    resultSection.style.display = "block";

    // 요약 카드
    const typeInfo = {
        "Open Defect (TSV-RDL)": { class: "open-tsv", label: "① Open (TSV-RDL)" },
        "Open Defect (Bump)": { class: "open-bump", label: "② Open (Bump)" },
        "Short Defect (Bump)": { class: "short-bump", label: "③ Short (Bump)" },
        "Void Formation (TSV)": { class: "void-tsv", label: "④ Void (TSV)" },
        "Incomplete Fill (TSV)": { class: "incomplete-fill", label: "⑤ Incomplete Fill" },
    };

    let cardsHTML = `
        <div class="summary-card total">
            <div class="count">${data.total_defects}</div>
            <div class="label">전체 결함 수</div>
        </div>`;

    for (const [type, info] of Object.entries(typeInfo)) {
        const count = data.defect_summary[type] || 0;
        cardsHTML += `
            <div class="summary-card ${info.class}">
                <div class="count">${count}</div>
                <div class="label">${info.label}</div>
            </div>`;
    }
    summaryCards.innerHTML = cardsHTML;

    // 이미지 — base64로 표시
    originalResult.src = `data:image/png;base64,${data.original_image_b64}`;
    analyzedResult.src = `data:image/png;base64,${data.result_image_b64}`;

    // 결함 테이블
    defectTable.innerHTML = "";
    if (data.defects.length === 0) {
        noDefects.style.display = "block";
        document.querySelector(".defect-table").style.display = "none";
    } else {
        noDefects.style.display = "none";
        document.querySelector(".defect-table").style.display = "table";

        data.defects.forEach((defect, i) => {
            const badgeClass = getBadgeClass(defect.type);
            const pct = Math.round(defect.confidence * 100);
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${i + 1}</td>
                <td><span class="defect-type-badge ${badgeClass}">${defect.type}</span></td>
                <td>(${defect.bbox[0]}, ${defect.bbox[1]})</td>
                <td>${defect.bbox[2]} × ${defect.bbox[3]}</td>
                <td>
                    <span class="confidence-bar"><span class="confidence-fill" style="width:${pct}%"></span></span>
                    ${pct}%
                </td>
                <td>${defect.description}</td>`;
            defectTable.appendChild(row);
        });
    }

    // 공정 원인 분석 표시
    if (data.process_analysis && data.total_defects > 0) {
        renderProcessAnalysis(data.process_analysis);
    } else {
        document.getElementById("processAnalysis").style.display = "none";
    }

    // 결과로 스크롤
    resultSection.scrollIntoView({ behavior: "smooth" });
}

// ===== 공정 원인 분석 렌더링 =====
function renderProcessAnalysis(analysis) {
    const section = document.getElementById("processAnalysis");
    section.style.display = "block";

    // 1) 공정 파이프라인
    const pipeline = document.getElementById("processPipeline");
    pipeline.innerHTML = analysis.stages.map((s) => {
        const relatedHTML = s.related_defects.length > 0
            ? `<div class="stage-related">${s.related_defects.join(", ")}</div>`
            : "";
        return `
            <div class="pipeline-stage">
                <div class="stage-circle risk-${s.risk_level}">${s.risk_pct}%</div>
                <div class="stage-name">${s.stage}</div>
                <div class="stage-label">${s.name}</div>
                <span class="stage-risk-tag risk-${s.risk_level}">${getRiskLabel(s.risk_level)}</span>
                ${relatedHTML}
            </div>`;
    }).join("");

    // 2) 결함-공정 매트릭스 테이블
    const matrixWrapper = document.getElementById("matrixWrapper");
    const matrixKeys = Object.keys(analysis.matrix);
    if (matrixKeys.length > 0) {
        matrixWrapper.style.display = "block";
        const stages = ["S1 Via", "S2 Liner", "S3 Cu Fill", "S4 CMP", "S5 Bond"];
        const thead = document.querySelector("#matrixTable thead");
        const tbody = document.querySelector("#matrixTable tbody");

        thead.innerHTML = `<tr><th>결함</th>${stages.map(s => `<th>${s}</th>`).join("")}</tr>`;
        tbody.innerHTML = matrixKeys.map((defect) => {
            const row = analysis.matrix[defect];
            const cells = stages.map((s) => {
                const val = row[s] || "";
                if (val === "high") return `<td><span class="matrix-dot-high"></span></td>`;
                if (val === "low") return `<td><span class="matrix-dot-low"></span></td>`;
                return `<td></td>`;
            }).join("");
            return `<tr><td>${defect}</td>${cells}</tr>`;
        }).join("");
    } else {
        matrixWrapper.style.display = "none";
    }

    // 3) 조치 가이드
    const actionGuide = document.getElementById("actionGuide");
    const actionEntries = Object.entries(analysis.actions);
    if (actionEntries.length > 0) {
        actionGuide.innerHTML = actionEntries.map(([type, actions]) => {
            const badgeClass = getBadgeClass(type);
            const listItems = actions.map(a => `<li>${a}</li>`).join("");
            return `
                <div class="action-card">
                    <h5><span class="action-badge ${badgeClass}">${getShortName(type)}</span> 권장 조치</h5>
                    <ul>${listItems}</ul>
                </div>`;
        }).join("");
    } else {
        actionGuide.innerHTML = "";
    }
}

function getRiskLabel(level) {
    const labels = { none: "양호", low: "낮음", medium: "주의", high: "위험" };
    return labels[level] || level;
}

function getShortName(type) {
    const map = {
        "Open Defect (TSV-RDL)": "Open(TSV-RDL)",
        "Open Defect (Bump)": "Open(Bump)",
        "Short Defect (Bump)": "Short(Bump)",
        "Void Formation (TSV)": "Void(TSV)",
        "Incomplete Fill (TSV)": "Incomplete Fill",
    };
    return map[type] || type;
}

function getBadgeClass(type) {
    const map = {
        "Open Defect (TSV-RDL)": "badge-open-tsv",
        "Open Defect (Bump)": "badge-open-bump",
        "Short Defect (Bump)": "badge-short-bump",
        "Void Formation (TSV)": "badge-void-tsv",
        "Incomplete Fill (TSV)": "badge-incomplete-fill",
    };
    return map[type] || "";
}

// ===== 리셋 =====
resetBtn.addEventListener("click", resetUI);
newAnalysisBtn.addEventListener("click", resetUI);

function resetUI() {
    selectedFile = null;
    fileInput.value = "";
    uploadArea.style.display = "block";
    previewSection.style.display = "none";
    loadingSection.style.display = "none";
    resultSection.style.display = "none";
    window.scrollTo({ top: 0, behavior: "smooth" });
}
