// QUESTIONS
const questions = [
    "Does the child make eye contact?",
    "Does the child respond to their name?",
    "Does the child point to objects of interest?",
    "Does the child engage in pretend play?",
    "Does the child show repetitive behaviors?",
    "Is the child sensitive to certain sounds or textures?",
    "Does the child prefer to play alone?",
    "Does the child use gestures while communicating?",
    "Is there any delay in speech development?",
    "Does the child show restricted or fixed interests?"
];

let currentIndex = 0;
let answers = {};

// ------------------------------
// Start Screening
// ------------------------------
function startScreening() {
    console.log("Screening started");

    document.getElementById("introSection").classList.add("hidden");
    document.getElementById("questionsSection").classList.remove("hidden");

    currentIndex = 0;
    answers = {};

    loadQuestion();
}

// ------------------------------
// Load Question
// ------------------------------
function loadQuestion() {
    const container = document.getElementById("questionsContainer");
    container.innerHTML = `
        <h2>${questions[currentIndex]}</h2>

        <div class="option-group">
            <button onclick="selectAnswer(1)" class="btn btn-primary">Yes</button>
            <button onclick="selectAnswer(0)" class="btn btn-secondary">No</button>
        </div>
    `;

    updateProgress();
}

// ------------------------------
// Save Answer
// ------------------------------
function selectAnswer(value) {
    answers[currentIndex] = value;

    if (currentIndex < questions.length - 1) {
        currentIndex++;
        loadQuestion();
    } else {
        document.getElementById("submitBtn").disabled = false;
    }
}

// ------------------------------
// Progress Bar Update
// ------------------------------
function updateProgress() {
    let percent = Math.round(((currentIndex + 1) / questions.length) * 100);

    document.getElementById("progressText").innerText =
        `Question ${currentIndex + 1} of ${questions.length}`;
    document.getElementById("progressPercent").innerText = `${percent}%`;
    document.getElementById("progressFill").style.width = percent + "%";
}

// ------------------------------
// Submit Answers
// ------------------------------
async function submitAnswers() {

    document.getElementById("questionsSection").classList.add("hidden");
    document.getElementById("loadingSection").classList.remove("hidden");

    // Convert answers into proper API fields
    const payload = {
        eye_contact: answers[0],
        responds_name: answers[1],
        points_objects: answers[2],
        pretend_play: answers[3],
        repetitive_behaviour: answers[4],
        sensory_sensitivity: answers[5],
        prefers_alone: answers[6],
        gestures: answers[7],
        delayed_speech: answers[8],
        restricted_interests: answers[9]
    };

    console.log("Payload sent:", payload);

    const result = await sendPrediction(payload);

    showResults(result);
}

// ------------------------------
// Show Results
// ------------------------------
function showResults(result) {
    document.getElementById("loadingSection").classList.add("hidden");
    document.getElementById("resultsSection").classList.remove("hidden");

    let html = `
        <h2>Screening Results</h2>
        <p><strong>Probability:</strong> ${result.probability}</p>
        <p><strong>Risk Level:</strong> ${result.risk_level}</p>
        <p><strong>Explanation:</strong> ${result.explanation}</p>
        <p><strong>Doctor Recommendation:</strong> ${result.doctor_recommendation}</p>

        <button class="btn btn-primary" onclick="resetScreening()">Start Again</button>
    `;

    document.getElementById("resultsSection").innerHTML = html;
}

// ------------------------------
// Reset
// ------------------------------
function resetScreening() {
    location.reload();
}
