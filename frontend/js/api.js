const API_BASE_URL = 'http://localhost:8000';

const apiService = {
    async getQuestions() {
        try {
            const response = await fetch(`${API_BASE_URL}/questions`);
            const data = await response.json();
            return { success: true, data: data.questions };
        } catch (error) {
            return { success: false, error: error.message };
        }
    },

    async submitScreening(answers) {
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(answers)
            });
            const data = await response.json();
            return { success: true, data };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
};