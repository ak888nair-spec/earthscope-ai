// ============================================================
// EARTHSCOPE AI
// Frontend Control System
// ============================================================


// ============================================================
// ELEMENT REFERENCES
// These IDs match the current index.html exactly.
// ============================================================
const API_URL = "https://earthscope-ai.onrender.com";
const uploadView = document.getElementById("uploadView");

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const chooseBtn = document.getElementById("chooseBtn");

const previewCard = document.getElementById("previewCard");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const fileMeta = document.getElementById("fileMeta");

const replaceBtn = document.getElementById("replaceBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

const loadingView = document.getElementById("loadingView");
const loadingTitle = document.getElementById("loadingTitle");
const loadingText = document.getElementById("loadingText");
const progressBar = document.getElementById("progressBar");

const resultView = document.getElementById("resultView");
const resultImage = document.getElementById("resultImage");
const toggleImageBtn = document.getElementById("toggleImageBtn");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");

const relevanceScore = document.getElementById("relevanceScore");
const featureCount = document.getElementById("featureCount");
const featureList = document.getElementById("featureList");
const explanation = document.getElementById("explanation");
const modelNote = document.getElementById("modelNote");
const legend = document.getElementById("legend");

const unsuitableView =
    document.getElementById("unsuitableView");

const unsuitableTitle =
    document.getElementById("unsuitableTitle");

const unsuitableMessage =
    document.getElementById("unsuitableMessage");

const retryUnsuitable =
    document.getElementById("retryUnsuitable");

const errorView =
    document.getElementById("errorView");

const errorMessage =
    document.getElementById("errorMessage");

const retryError =
    document.getElementById("retryError");

const newFromError =
    document.getElementById("newFromError");


// Loading pipeline elements
const pipe1 = document.getElementById("pipe1");
const pipe2 = document.getElementById("pipe2");
const pipe3 = document.getElementById("pipe3");
const pipe4 = document.getElementById("pipe4");
const pipe5 = document.getElementById("pipe5");


// ============================================================
// APPLICATION STATE
// ============================================================

let selectedFile = null;

let originalImageData = null;
let annotatedImageData = null;

let showingOriginal = false;

let loadingTimer = null;


// ============================================================
// SECURITY / HTML ESCAPING
// ============================================================

function escapeHTML(value) {

    if (
        value === null ||
        value === undefined
    ) {
        return "";
    }

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


// ============================================================
// VIEW MANAGEMENT
// ============================================================

function hideAllViews() {

    uploadView.classList.add("hidden");
    previewCard.classList.add("hidden");
    loadingView.classList.add("hidden");
    resultView.classList.add("hidden");
    unsuitableView.classList.add("hidden");
    errorView.classList.add("hidden");
}


function showUploadView() {

    hideAllViews();

    uploadView.classList.remove("hidden");

    dropZone.classList.remove("hidden");
    previewCard.classList.add("hidden");
}


function showPreviewView() {

    hideAllViews();

    uploadView.classList.remove("hidden");

    dropZone.classList.add("hidden");
    previewCard.classList.remove("hidden");
}


function showLoadingView() {

    hideAllViews();

    loadingView.classList.remove("hidden");
}


function showResultView() {

    hideAllViews();

    resultView.classList.remove("hidden");
}


function showUnsuitableView() {

    hideAllViews();

    unsuitableView.classList.remove("hidden");
}


function showErrorView() {

    hideAllViews();

    errorView.classList.remove("hidden");
}


// ============================================================
// INITIAL STATE
// ============================================================

showUploadView();


// ============================================================
// FILE PICKER
// ============================================================

chooseBtn.addEventListener(
    "click",
    function(event) {

        // Important:
        // Prevent the button click from triggering
        // the label/drop-zone click twice.
        event.preventDefault();
        event.stopPropagation();

        fileInput.click();
    }
);


fileInput.addEventListener(
    "change",
    function() {

        if (
            !fileInput.files ||
            fileInput.files.length === 0
        ) {
            return;
        }

        const file =
            fileInput.files[0];

        selectFile(file);
    }
);


// ============================================================
// SELECT FILE
// ============================================================

function selectFile(file) {

    if (!file) {
        return;
    }


    // --------------------------------------------------------
    // File type check
    // --------------------------------------------------------

    if (!file.type.startsWith("image/")) {

        showError(
            "INVALID PAYLOAD // Please select a valid image file."
        );

        return;
    }


    // --------------------------------------------------------
    // File size check
    // --------------------------------------------------------

    const maxSize =
        15 * 1024 * 1024;

    if (file.size > maxSize) {

        showError(
            "PAYLOAD TOO LARGE // Maximum supported image size is 15 MB."
        );

        return;
    }


    // --------------------------------------------------------
    // Store file
    // --------------------------------------------------------

    selectedFile = file;


    // --------------------------------------------------------
    // Display metadata
    // --------------------------------------------------------

    fileName.textContent =
        file.name;

    fileMeta.textContent =
        `${formatFileSize(file.size)} // ${file.type || "IMAGE"}`;


    // --------------------------------------------------------
    // Preview
    // --------------------------------------------------------

    const reader =
        new FileReader();


    reader.onload =
        function(event) {

            previewImage.src =
                event.target.result;

            showPreviewView();

        };


    reader.onerror =
        function() {

            showError(
                "PAYLOAD READ ERROR // The selected image could not be read."
            );
        };


    reader.readAsDataURL(file);
}


// ============================================================
// FILE SIZE FORMATTER
// ============================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {
        return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {

        return `${(
            bytes / 1024
        ).toFixed(1)} KB`;
    }

    return `${(
        bytes / (
            1024 * 1024
        )
    ).toFixed(2)} MB`;
}


// ============================================================
// DRAG & DROP
// ============================================================

dropZone.addEventListener(
    "dragover",
    function(event) {

        event.preventDefault();

        dropZone.classList.add(
            "dragging"
        );
    }
);


dropZone.addEventListener(
    "dragenter",
    function(event) {

        event.preventDefault();

        dropZone.classList.add(
            "dragging"
        );
    }
);


dropZone.addEventListener(
    "dragleave",
    function(event) {

        event.preventDefault();

        dropZone.classList.remove(
            "dragging"
        );
    }
);


dropZone.addEventListener(
    "drop",
    function(event) {

        event.preventDefault();

        event.stopPropagation();

        dropZone.classList.remove(
            "dragging"
        );


        const files =
            event.dataTransfer.files;


        if (
            !files ||
            files.length === 0
        ) {
            return;
        }


        selectFile(
            files[0]
        );
    }
);


// ============================================================
// REPLACE IMAGE
// ============================================================

replaceBtn.addEventListener(
    "click",
    function(event) {

        event.preventDefault();

        fileInput.click();
    }
);


// ============================================================
// LOADING PIPELINE
// ============================================================

function resetPipeline() {

    const pipes = [
        pipe1,
        pipe2,
        pipe3,
        pipe4,
        pipe5
    ];


    pipes.forEach(
        function(pipe) {

            pipe.classList.remove(
                "active"
            );

            pipe.classList.remove(
                "complete"
            );
        }
    );


    pipe1.classList.add(
        "active"
    );


    progressBar.style.width =
        "5%";
}


function startLoadingAnimation() {

    resetPipeline();

    showLoadingView();


    const stages = [
        {
            pipe: pipe1,
            title: "Receiving observation",
            text: "Reading the uploaded image payload.",
            progress: 12
        },

        {
            pipe: pipe2,
            title: "Checking image quality",
            text: "Evaluating resolution, contrast and visual clarity.",
            progress: 30
        },

        {
            pipe: pipe3,
            title: "Running planetary vision",
            text: "Comparing the observation against Earth phenomena.",
            progress: 52
        },

        {
            pipe: pipe4,
            title: "Mapping visible features",
            text: "Locating clouds, terrain, water and other visual regions.",
            progress: 75
        },

        {
            pipe: pipe5,
            title: "Building observation",
            text: "Separating visual evidence from interpretation.",
            progress: 94
        }
    ];


    let index = 0;


    function advanceStage() {

        if (index > 0) {

            stages[index - 1]
                .pipe
                .classList.remove(
                    "active"
                );

            stages[index - 1]
                .pipe
                .classList.add(
                    "complete"
                );
        }


        if (
            index < stages.length
        ) {

            const stage =
                stages[index];


            stage.pipe.classList.add(
                "active"
            );


            loadingTitle.textContent =
                stage.title;


            loadingText.textContent =
                stage.text;


            progressBar.style.width =
                `${stage.progress}%`;


            index++;

        } else {

            clearInterval(
                loadingTimer
            );
        }
    }


    advanceStage();


    loadingTimer =
        setInterval(
            advanceStage,
            900
        );
}


// ============================================================
// STOP LOADING ANIMATION
// ============================================================

function stopLoadingAnimation() {

    if (loadingTimer !== null) {

        clearInterval(
            loadingTimer
        );

        loadingTimer = null;
    }


    const pipes = [
        pipe1,
        pipe2,
        pipe3,
        pipe4,
        pipe5
    ];


    pipes.forEach(
        function(pipe) {

            pipe.classList.remove(
                "active"
            );

            pipe.classList.add(
                "complete"
            );
        }
    );


    progressBar.style.width =
        "100%";
}


// ============================================================
// ANALYZE BUTTON
// ============================================================

analyzeBtn.addEventListener(
    "click",
    function() {

        analyzeImage();
    }
);


// ============================================================
// ANALYZE IMAGE
// ============================================================

async function analyzeImage() {

    if (!selectedFile) {

        showError(
            "NO PAYLOAD // Select an image before initiating analysis."
        );

        return;
    }


    startLoadingAnimation();


    const formData =
        new FormData();


    formData.append(
        "image",
        selectedFile
    );


    try {

        const response =
            await fetch(
                `${API_URL}/api/analyze`,
                {
                    method: "POST",
                    body: formData
                }
            );


        let data;


        try {

            data =
                await response.json();

        } catch (jsonError) {

            throw new Error(
                "The analysis engine returned an invalid response."
            );
        }


        stopLoadingAnimation();


        // ----------------------------------------------------
        // Server error
        // ----------------------------------------------------

        if (!response.ok) {

            showError(
                data.message ||
                "The analysis engine returned a server error."
            );

            return;
        }


        // ----------------------------------------------------
        // Route result
        // ----------------------------------------------------

        handleAnalysisResult(
            data
        );

    }

    catch (error) {

        console.error(
            "EARTHSCOPE ANALYSIS ERROR:",
            error
        );


        stopLoadingAnimation();


        showError(
            error.message ||
            "EarthScope could not communicate with the analysis engine."
        );
    }
}


// ============================================================
// RESULT ROUTER
// ============================================================

function handleAnalysisResult(data) {

    switch (data.status) {

        case "success":

            renderSuccess(
                data
            );

            break;


        case "no_features":

            renderNoFeatures(
                data
            );

            break;


        case "unsuitable":

            renderUnsuitable(
                data
            );

            break;


        case "invalid":

            showError(
                data.message ||
                "The uploaded payload is invalid."
            );

            break;


        case "error":

            showError(
                data.message ||
                "The observation engine encountered an error."
            );

            break;


        default:

            showError(
                "UNKNOWN SYSTEM RESPONSE // EarthScope received an unsupported response."
            );
    }
}


// ============================================================
// SUCCESS RESULT
// ============================================================

function renderSuccess(data) {

    showResultView();


    // --------------------------------------------------------
    // Store images
    // --------------------------------------------------------

    originalImageData =
        data.original_image;

    annotatedImageData =
        data.annotated_image;


    showingOriginal = false;


    // --------------------------------------------------------
    // Display annotated image
    // --------------------------------------------------------

    resultImage.src =
        annotatedImageData;


    toggleImageBtn.textContent =
        "SHOW ORIGINAL";


    // --------------------------------------------------------
    // Relevance
    // --------------------------------------------------------

    relevanceScore.textContent =
        `RELEVANCE — ${data.relevance}%`;


    // --------------------------------------------------------
    // Feature count
    // --------------------------------------------------------

    featureCount.textContent =
        String(
            data.detections.length
        ).padStart(
            2,
            "0"
        );


    // --------------------------------------------------------
    // Feature cards
    // --------------------------------------------------------

    renderFeatures(
        data.detections
    );


    // --------------------------------------------------------
    // Legend
    // --------------------------------------------------------

    renderLegend(
        data.detections
    );


    // --------------------------------------------------------
    // Explanation
    // --------------------------------------------------------

    explanation.textContent =
        data.explanation ||
        "No explanation was generated.";


    // --------------------------------------------------------
    // System note
    // --------------------------------------------------------

    modelNote.textContent =
        data.model_note ||
        "Results are AI-generated visual estimates.";
}


// ============================================================
// FEATURE RENDERING
// ============================================================

function renderFeatures(
    detections
) {

    featureList.innerHTML =
        "";


    if (
        !detections ||
        detections.length === 0
    ) {

        featureList.innerHTML = `
            <div class="feature-empty">
                NO HIGH-CONFIDENCE FEATURES DETECTED
            </div>
        `;

        return;
    }


    detections.forEach(
        function(feature, index) {

            const confidence =
                Number(
                    feature.confidence || 0
                );


            let areaText;


            if (
                feature.area_percent !== null &&
                feature.area_percent !== undefined
            ) {

                areaText =
                    `${feature.area_percent}% area`;

            } else {

                areaText =
                    "image-level detection";
            }


            let typeLabel;


            if (
                feature.type === "phenomenon"
            ) {

                typeLabel =
                    "PHENOMENON";

            } else {

                typeLabel =
                    "VISIBLE REGION";
            }


            const item =
                document.createElement(
                    "div"
                );


            item.className =
                "feature-item";


            item.innerHTML = `
                <div class="feature-index">
                    ${String(index + 1).padStart(2, "0")}
                </div>

                <div class="feature-content">

                    <div class="feature-type">
                        ${typeLabel}
                    </div>

                    <div class="feature-name">
                        ${escapeHTML(feature.name)}
                    </div>

                    <div class="feature-description">
                        ${escapeHTML(feature.description)}
                    </div>

                    <div class="confidence-track">

                        <div
                            class="confidence-fill"
                            style="width: ${confidence}%"
                        ></div>

                    </div>

                    <div class="feature-meta">
                        ${confidence.toFixed(1)}% visual confidence
                        &nbsp;•&nbsp;
                        ${escapeHTML(areaText)}
                    </div>

                </div>
            `;


            featureList.appendChild(
                item
            );
        }
    );
}


// ============================================================
// LEGEND
// ============================================================

function renderLegend(
    detections
) {

    legend.innerHTML =
        "";


    if (
        !detections ||
        detections.length === 0
    ) {
        return;
    }


    detections.forEach(
        function(feature) {

            const item =
                document.createElement(
                    "div"
                );


            item.className =
                "legend-item";


            let symbol;


            if (
                feature.type === "phenomenon"
            ) {

                symbol =
                    "◆";

            } else {

                symbol =
                    "□";
            }


            item.innerHTML = `
                <span class="legend-symbol">
                    ${symbol}
                </span>

                <span>
                    ${escapeHTML(feature.name)}
                </span>
            `;


            legend.appendChild(
                item
            );
        }
    );
}


// ============================================================
// NO HIGH-CONFIDENCE FEATURES
// ============================================================

function renderNoFeatures(
    data
) {

    showResultView();


    originalImageData =
        data.original_image;

    annotatedImageData =
        data.annotated_image;


    showingOriginal = false;


    resultImage.src =
        annotatedImageData;


    toggleImageBtn.textContent =
        "SHOW ORIGINAL";


    relevanceScore.textContent =
        `RELEVANCE — ${data.relevance}%`;


    featureCount.textContent =
        "00";


    featureList.innerHTML = `
        <div class="feature-item">

            <div class="feature-index">
                --
            </div>

            <div class="feature-content">

                <div class="feature-type">
                    NO HIGH-CONFIDENCE DETECTION
                </div>

                <div class="feature-name">
                    Observation inconclusive
                </div>

                <div class="feature-description">
                    ${escapeHTML(data.message)}
                </div>

            </div>

        </div>
    `;


    legend.innerHTML =
        "";


    explanation.textContent =
        data.explanation ||
        data.message ||
        "";


    modelNote.textContent =
        data.model_note ||
        "The system did not obtain enough evidence for a supported feature.";
}


// ============================================================
// UNSUITABLE IMAGE
// ============================================================

function renderUnsuitable(
    data
) {

    showUnsuitableView();


    let title =
        "Observation unsuitable.";


    switch (
        data.reason
    ) {

        case "blurry":

            title =
                "Observation too blurred.";

            break;


        case "low_resolution":

            title =
                "Resolution insufficient.";

            break;


        case "low_information":

            title =
                "Insufficient visual information.";

            break;


        case "unrelated":

            title =
                "Observation not relevant.";

            break;
    }


    unsuitableTitle.textContent =
        title;


    unsuitableMessage.textContent =
        data.message ||
        "The uploaded image could not pass the quality gate.";
}


// ============================================================
// ERROR
// ============================================================

function showError(
    message
) {

    showErrorView();


    errorMessage.textContent =
        message ||
        "An unexpected system error occurred.";
}


// ============================================================
// IMAGE TOGGLE
// ============================================================

toggleImageBtn.addEventListener(
    "click",
    function() {

        if (
            !originalImageData ||
            !annotatedImageData
        ) {
            return;
        }


        showingOriginal =
            !showingOriginal;


        if (showingOriginal) {

            resultImage.src =
                originalImageData;


            toggleImageBtn.textContent =
                "SHOW DETECTION";

        } else {

            resultImage.src =
                annotatedImageData;


            toggleImageBtn.textContent =
                "SHOW ORIGINAL";
        }
    }
);


// ============================================================
// NEW ANALYSIS
// ============================================================

newAnalysisBtn.addEventListener(
    "click",
    function() {

        resetApplication();

    }
);


// ============================================================
// RETRY FROM UNSUITABLE
// ============================================================

retryUnsuitable.addEventListener(
    "click",
    function() {

        resetApplication();

    }
);


// ============================================================
// RETRY ERROR
// ============================================================

retryError.addEventListener(
    "click",
    function() {

        if (selectedFile) {

            analyzeImage();

        } else {

            resetApplication();
        }
    }
);


// ============================================================
// NEW IMAGE FROM ERROR
// ============================================================

newFromError.addEventListener(
    "click",
    function() {

        resetApplication();

    }
);


// ============================================================
// RESET APPLICATION
// ============================================================

function resetApplication() {

    selectedFile = null;

    originalImageData = null;
    annotatedImageData = null;

    showingOriginal = false;


    if (loadingTimer !== null) {

        clearInterval(
            loadingTimer
        );

        loadingTimer = null;
    }


    fileInput.value =
        "";


    previewImage.removeAttribute(
        "src"
    );


    fileName.textContent =
        "";


    fileMeta.textContent =
        "";


    resultImage.removeAttribute(
        "src"
    );


    featureList.innerHTML =
        "";


    legend.innerHTML =
        "";


    explanation.textContent =
        "";


    modelNote.textContent =
        "";


    relevanceScore.textContent =
        "RELEVANCE — --";


    featureCount.textContent =
        "00";


    progressBar.style.width =
        "0%";


    resetPipeline();


    showUploadView();


    // Scroll back to the upload system.
    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}


// ============================================================
// KEYBOARD SUPPORT
// ============================================================

document.addEventListener(
    "keydown",
    function(event) {

        // Enter while preview is active
        // initiates analysis.
        if (
            event.key === "Enter" &&
            !previewCard.classList.contains("hidden") &&
            selectedFile
        ) {

            analyzeImage();
        }
    }
);


// ============================================================
// DEBUG INFORMATION
// ============================================================

console.log(
    "%cEARTHSCOPE AI",
    "color:#4deaff;font-weight:bold;font-size:18px;"
);

console.log(
    "%cPlanetary observation interface initialized.",
    "color:#8aa0aa;"
);

console.log(
    "Upload system:",
    !!fileInput
);

console.log(
    "Analysis system:",
    !!analyzeBtn
);

console.log(
    "Result system:",
    !!resultView
);