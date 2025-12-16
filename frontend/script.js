// script.js

const API_BASE = "http://127.0.0.1:8001";

/* ----------------------------
   AUTH HELPERS
----------------------------- */
function getToken() {
  return localStorage.getItem("access_token");
}

function getAuthHeaders() {
  const token = getToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

/* ----------------------------
   DASHBOARD DATA LOADING
----------------------------- */
async function loadDashboard() {
  try {
    const res = await fetch(`${API_BASE}/dashboard/summary`, {
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      },
    });

    if (!res.ok) {
      console.error("Failed to load dashboard summary");
      return;
    }

    const data = await res.json();
    console.log("Dashboard summary:", data);

    // Stats
    const totalEl = document.getElementById("total-leaves");
    const healthyEl = document.getElementById("healthy-leaves");
    const diseasedEl = document.getElementById("diseased-leaves");

    if (totalEl) totalEl.textContent = data.total_leaves_scanned ?? "-";
    if (healthyEl) healthyEl.textContent = data.healthy_leaves ?? "-";
    if (diseasedEl) diseasedEl.textContent = data.diseased_leaves ?? "-";

    // Farm summary text + PDF link
    const summaryTextEl = document.getElementById("farm-summary-text");
    const reportLinkEl = document.getElementById("report-link");

    if (summaryTextEl) {
      summaryTextEl.textContent = `Farm: ${data.farm_name} • Last scan: ${data.last_scan_time}`;
    }

    if (reportLinkEl && data.report_url) {
      reportLinkEl.href = `${API_BASE}${data.report_url}`;
      reportLinkEl.target = "_blank";
    }

    // Drone info
    const droneStatusEl = document.getElementById("drone-status");
    const droneBattEl = document.getElementById("drone-battery");
    const droneFlightEl = document.getElementById("drone-last-flight");

    if (droneStatusEl) {
      droneStatusEl.textContent = `Status: ${data.drone_status}`;
      droneStatusEl.classList.toggle("online", data.drone_status === "Online");
      droneStatusEl.classList.toggle("offline", data.drone_status !== "Online");
    }
    if (droneBattEl) droneBattEl.textContent = `Battery: ${data.battery_level}%`;
    if (droneFlightEl) {
      droneFlightEl.textContent = `Last flight: ${data.last_flight_duration_min} min`;
    }

    // User info
    const userNameEl = document.getElementById("user-name");
    const userEmailEl = document.getElementById("user-email");

    if (userNameEl) {
      userNameEl.textContent = localStorage.getItem("user_name") || "Farmer";
    }
    if (userEmailEl) {
      userEmailEl.textContent = "farmer@example.com";
    }
  } catch (err) {
    console.error("Error loading dashboard:", err);
  }
}

/* ----------------------------
   DRONE SCAN BUTTON
----------------------------- */
async function startDroneScan() {
  const btn = document.getElementById("btn-start-scan");
  try {
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Scanning…";
    }

    const res = await fetch(`${API_BASE}/drone/start_scan`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      },
    });

    if (!res.ok) {
      alert("Failed to start drone scan.");
      return;
    }

    const data = await res.json();
    console.log("Drone scan response:", data);

    // Update stats
    const totalEl = document.getElementById("total-leaves");
    const healthyEl = document.getElementById("healthy-leaves");
    const diseasedEl = document.getElementById("diseased-leaves");

    if (totalEl) totalEl.textContent = data.total_leaves_scanned ?? "-";
    if (healthyEl) healthyEl.textContent = data.healthy_leaves ?? "-";
    if (diseasedEl) diseasedEl.textContent = data.diseased_leaves ?? "-";

    // Drone info
    const droneStatusEl = document.getElementById("drone-status");
    const droneBattEl = document.getElementById("drone-battery");
    const droneFlightEl = document.getElementById("drone-last-flight");

    if (droneStatusEl) {
      droneStatusEl.textContent = `Status: ${data.drone_status}`;
      droneStatusEl.classList.toggle("online", data.drone_status === "Online");
      droneStatusEl.classList.toggle("offline", data.drone_status !== "Online");
    }
    if (droneBattEl) droneBattEl.textContent = `Battery: ${data.battery_level}%`;
    if (droneFlightEl) {
      droneFlightEl.textContent = `Last flight: ${data.last_flight_duration_min} min`;
    }

    alert("Drone scan simulated and dashboard updated.");
  } catch (err) {
    console.error("Error starting scan:", err);
    alert("Error starting scan.");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Start Drone Scan";
    }
  }
}

/* ----------------------------
   CHATBOT (Backend-connected)
----------------------------- */
function setupChatbot() {
  const input = document.getElementById("chat-input");
  const historyBox = document.getElementById("chat-history");

  if (!input || !historyBox) return;

  input.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && input.value.trim()) {
      const userText = input.value.trim();

      addMessage(historyBox, userText, "user");
      input.value = "";

      await sendChatMessage(historyBox, userText);
    }
  });
}

function addMessage(historyBox, text, type) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${type}`;
  bubble.textContent = text;
  historyBox.appendChild(bubble);
  historyBox.scrollTop = historyBox.scrollHeight;
}

async function sendChatMessage(historyBox, userText) {
  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      },
      body: JSON.stringify({ message: userText }),
    });

    if (!res.ok) {
      console.error("Chat request failed:", res.status);
      addMessage(historyBox, "Assistant unavailable right now.", "bot");
      return;
    }

    const data = await res.json();
    addMessage(historyBox, data.reply ?? "...", "bot");
  } catch (err) {
    console.error("Chat error:", err);
    addMessage(historyBox, "Network error. Try again.", "bot");
  }
}

/* ----------------------------
   DISEASE TREND CHART
----------------------------- */
async function loadHistoryChart() {
  const canvas = document.getElementById("disease-chart");
  if (!canvas) return;
  if (typeof Chart === "undefined") {
    console.error("Chart.js not loaded");
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/dashboard/history`, {
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders(),
      },
    });

    if (!res.ok) {
      console.error("Failed to load history:", res.status);
      return;
    }

    const data = await res.json();
    const ctx = canvas.getContext("2d");

    new Chart(ctx, {
      type: "line",
      data: {
        labels: data.days,
        datasets: [
          {
            label: "Diseased Leaves",
            data: data.diseased_leaves,
            borderColor: "#d63031",
            borderWidth: 2,
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: true },
        },
      },
    });
  } catch (err) {
    console.error("Chart load failed:", err);
  }
}

/* ----------------------------
   IMAGE PREDICTION
----------------------------- */
async function handlePredictClick() {
  const fileInput = document.getElementById("leaf-file");
  const resultBox = document.getElementById("predict-result");

  if (!fileInput || !resultBox) return;

  if (!fileInput.files || fileInput.files.length === 0) {
    resultBox.textContent = "Please select an image first.";
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  resultBox.textContent = "Analyzing leaf image…";

  try {
    const res = await fetch(`${API_BASE}/predict/image`, {
      method: "POST",
      headers: {
        // DO NOT set Content-Type manually here
        ...getAuthHeaders(),
      },
      body: formData,
    });

    if (!res.ok) {
      resultBox.textContent = "Prediction failed. Please try again.";
      console.error("Predict error:", res.status);
      return;
    }

    const data = await res.json();

    resultBox.innerHTML = `
      <strong>Predicted class:</strong> ${data.predicted_class}<br/>
      <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%<br/>
      <strong>Severity:</strong> ${data.severity}<br/>
      <strong>Recommendation:</strong> ${data.recommendation}
    `;
  } catch (err) {
    console.error("Predict error:", err);
    resultBox.textContent = "Network error while sending image.";
  }
}

/* ----------------------------
   INIT
----------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  loadDashboard();
  loadHistoryChart();

  const startBtn = document.getElementById("btn-start-scan");
  if (startBtn) {
    startBtn.addEventListener("click", startDroneScan);
  }

  const predictBtn = document.getElementById("btn-predict");
  if (predictBtn) {
    predictBtn.addEventListener("click", handlePredictClick);
  }

  setupChatbot();
});
