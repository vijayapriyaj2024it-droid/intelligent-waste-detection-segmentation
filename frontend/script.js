// Flask Backend Server API Base URL
const API_BASE_URL = "http://127.0.0.1:5000";

// App State
let currentSelectedFile = null;
let currentAnalysisResponse = null;

// DOM Element Selectors
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const dropZonePrompt = document.getElementById("dropZonePrompt");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");

const fileDetails = document.getElementById("fileDetails");
const fileName = document.getElementById("fileName");
const fileDimensions = document.getElementById("fileDimensions");
const removeBtn = document.getElementById("removeBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

const loadingSpinner = document.getElementById("loadingSpinner");
const resultSection = document.getElementById("resultSection");

const resultOriginalImg = document.getElementById("resultOriginalImg");
const resultSegmentationImg = document.getElementById("resultSegmentationImg");
const detectedClass = document.getElementById("detectedClass");
const confidenceScore = document.getElementById("confidenceScore");
const objectsDetected = document.getElementById("objectsDetected");
const disposalRecommendation = document.getElementById(
    "disposalRecommendation"
);
const downloadBtn = document.getElementById("downloadBtn");
const resetBtn = document.getElementById("resetBtn");


// 1. Drag & Drop Event Listeners

dropZone.addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        processSelectedFile(e.target.files[0]);
    }
});

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");

    if (e.dataTransfer.files.length > 0) {
        processSelectedFile(e.dataTransfer.files[0]);
    }
});


// 2. Image File Selection Handler

function processSelectedFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {
        alert("Please select a valid image file (.jpg, .jpeg, or .png)");
        return;
    }

    currentSelectedFile = file;
    fileName.textContent = file.name;

    const reader = new FileReader();

    reader.onload = function (e) {

        previewImage.src = e.target.result;

        dropZonePrompt.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        fileDetails.classList.remove("hidden");

        const img = new Image();

        img.src = e.target.result;

        img.onload = function () {
            fileDimensions.textContent =
                `${this.width} × ${this.height} px`;
        };

        analyzeBtn.disabled = false;
    };

    reader.readAsDataURL(file);
}


// 3. Remove Uploaded File

removeBtn.addEventListener("click", (e) => {

    e.stopPropagation();

    resetUploader();
});


function resetUploader() {

    currentSelectedFile = null;
    currentAnalysisResponse = null;

    fileInput.value = "";
    previewImage.src = "";

    previewContainer.classList.add("hidden");
    dropZonePrompt.classList.remove("hidden");
    fileDetails.classList.add("hidden");

    analyzeBtn.disabled = true;

    resultSection.classList.add("hidden");
}


// 4. Send Image to Flask Backend

analyzeBtn.addEventListener("click", async () => {

    if (!currentSelectedFile) {
        alert("Please select an image first.");
        return;
    }

    loadingSpinner.classList.remove("hidden");
    resultSection.classList.add("hidden");
    analyzeBtn.disabled = true;

    const formData = new FormData();

    formData.append("image", currentSelectedFile);

    try {

        // IMPORTANT:
        // Flask backend endpoint is /predict
        const response = await fetch(
            `${API_BASE_URL}/predict`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.error || `Server error ${response.status}`
            );
        }

        currentAnalysisResponse = data;

        renderResults(data);

    } catch (err) {

        console.error("Analysis error:", err);

        alert(`Analysis Error: ${err.message}`);

    } finally {

        loadingSpinner.classList.add("hidden");
        analyzeBtn.disabled = false;
    }
});


// 5. Render Results

function renderResults(data) {

    // Original image
    resultOriginalImg.src = previewImage.src;


    // No objects detected
    if (!data.objects || data.objects.length === 0) {

        detectedClass.textContent = "No waste detected";

        confidenceScore.textContent = "0%";

        objectsDetected.textContent = "0";

        disposalRecommendation.textContent =
            data.recommendation ||
            "No waste objects were detected.";

    }

    // Objects detected
    else {

        const primary = data.objects.reduce(
            (best, current) =>
                current.confidence > best.confidence
                    ? current
                    : best
        );

        detectedClass.textContent =
            primary.category;

        confidenceScore.textContent =
            `${(primary.confidence * 100).toFixed(1)}%`;

        objectsDetected.textContent =
            data.total_items;

        disposalRecommendation.textContent =
            data.recommendation ||
            "Check the appropriate waste disposal stream.";
    }


    // Segmentation overlay
    const overlayImage =
        formatBase64(data.overlay_image);

    resultSegmentationImg.src =
        overlayImage;


    // Download button
    downloadBtn.href =
        overlayImage;

    downloadBtn.download =
        "waste_segmentation_result.png";


    // Show results
    resultSection.classList.remove("hidden");

    resultSection.scrollIntoView({
        behavior: "smooth"
    });
}


// 6. Convert Base64 Image

function formatBase64(base64String) {

    if (!base64String) {
        return "";
    }

    if (base64String.startsWith("data:image")) {
        return base64String;
    }

    return `data:image/png;base64,${base64String}`;
}


// 7. Segmentation View Toggle

document.querySelectorAll(".tab-btn").forEach((btn) => {

    btn.addEventListener("click", (e) => {

        document
            .querySelectorAll(".tab-btn")
            .forEach((button) => {
                button.classList.remove("active");
            });

        e.target.classList.add("active");

        if (!currentAnalysisResponse) {
            return;
        }


        const tabType =
            e.target.dataset.tab;


        // Overlay
        if (tabType === "overlay") {

            const overlay =
                formatBase64(
                    currentAnalysisResponse.overlay_image
                );

            resultSegmentationImg.src =
                overlay;
        }


        // Binary mask
        else if (tabType === "mask") {

            const mask =
                formatBase64(
                    currentAnalysisResponse.mask_image
                );

            resultSegmentationImg.src =
                mask;
        }
    });
});


// 8. Reset Dashboard

resetBtn.addEventListener("click", () => {

    resetUploader();

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
});