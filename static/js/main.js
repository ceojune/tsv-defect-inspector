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

    // 결과로 스크롤
    resultSection.scrollIntoView({ behavior: "smooth" });
}

function getBadgeClass(type) {
    const map = {
        "Open Defect (TSV-RDL)": "badge-open-tsv",
        "Open Defect (Bump)": "badge-open-bump",
        "Short Defect (Bump)": "badge-short-bump",
        "Void Formation (TSV)": "badge-void-tsv",
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
