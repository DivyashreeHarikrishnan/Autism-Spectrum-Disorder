let questions = [];
let answers = {};

async function startScreening() {
    Utils.hide('introSection');
    Utils.show('loadingSection');
    
    const result = await apiService.getQuestions();
    
    if (result.success) {
        questions = result.data;
        answers = {};
        renderQuestions();
        Utils.hide('loadingSection');
        Utils.show('questionsSection');
    } else {
        alert('Failed to load questions. Is the backend running?');
        Utils.hide('loadingSection');
        Utils.show('introSection');
    }
}

function renderQuestions() {
    const container = document.getElementById('questionsContainer');
    container.innerHTML = '';
    
    questions.forEach((q) => {
        const card = document.createElement('div');
        card.className = 'question-card';
        card.innerHTML = `
            <div class="question-number">Question ${q.id} of ${questions.length}</div>
            <div class="question-text">${q.text}</div>
            <div class="options">
                <button class="option-btn" onclick="selectAnswer('${q.field}', 1, this)">Yes</button>
                <button class="option-btn" onclick="selectAnswer('${q.field}', 0, this)">No</button>
            </div>
        `;
        container.appendChild(card);
    });
    
    updateProgress();
}

function selectAnswer(field, value, button) {
    answers[field] = value;
    
    const card = button.closest('.question-card');
    card.querySelectorAll('.option-btn').forEach(btn => btn.classList.remove('selected'));
    button.classList.add('selected');
    
    updateProgress();
    
    document.getElementById('submitBtn').disabled = Object.keys(answers).length !== questions.length;
}

function updateProgress() {
    const answered = Object.keys(answers).length;
    const total = questions.length;
    const percent = (answered / total) * 100;
    
    document.getElementById('progressFill').style.width = percent + '%';
    document.getElementById('progressText').textContent = `Question ${answered} of ${total}`;
    document.getElementById('progressPercent').textContent = `${Math.round(percent)}%`;
}

async function submitAnswers() {
    Utils.hide('questionsSection');
    Utils.show('loadingSection');
    
    const result = await apiService.submitScreening(answers);
    
    Utils.hide('loadingSection');
    
    if (result.success) {
        displayResults(result.data);
    } else {
        alert('Prediction failed: ' + result.error);
        Utils.show('questionsSection');
    }
}

function displayResults(result) {
    const riskClass = `risk-${result.risk_level.toLowerCase()}`;
    const icon = Utils.getRiskIcon(result.risk_level);
    
    document.getElementById('resultsSection').innerHTML = `
        <div class="result-card">
            <div class="result-icon ${riskClass}">${icon}</div>
            <h2 class="result-title">Screening Complete</h2>
            <p class="result-message">${result.message}</p>

            <div class="result-stats">
                <div class="stat-box">
                    <div class="stat-label">Risk Level</div>
                    <div class="stat-value ${riskClass}">${result.risk_level}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Confidence</div>
                    <div class="stat-value">${result.confidence}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Assessment</div>
                    <div class="stat-value">${result.prediction}</div>
                </div>
            </div>

            <div class="recommendation-box ${result.risk_level === 'High' ? 'high' : ''}">
                <strong>👨‍⚕️ Medical Recommendation:</strong><br><br>
                ${result.doctor_recommendation}
            </div>

            <button class="btn btn-primary" onclick="resetScreening()">Take Another Screening</button>
        </div>
    `;
    
    Utils.show('resultsSection');
}

function resetScreening() {
    answers = {};
    Utils.hide('questionsSection');
    Utils.hide('resultsSection');
    Utils.show('introSection');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}