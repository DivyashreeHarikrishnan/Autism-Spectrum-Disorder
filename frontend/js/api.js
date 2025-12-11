async function sendPrediction(data) {
    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        return await res.json();
    } catch (error) {
        console.error("API Error:", error);
        return { error: "Unable to connect to the server" };
    }
}
