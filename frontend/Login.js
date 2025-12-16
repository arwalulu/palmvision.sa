// Login.js

const API_BASE = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("login-form");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const email = emailInput.value.trim();
    const password = passwordInput.value.trim();

    if (!email || !password) return;

    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (!res.ok) {
        alert("Invalid email or password.");
        return;
      }

      const data = await res.json();

      // Save token + user name for dashboard
      localStorage.setItem("access_token", data.access_token);
      localStorage.setItem("user_name", data.user_name || "Farmer");

      // Go to dashboard
      window.location.href = "index.html";
    } catch (err) {
      console.error("Login error:", err);
      alert("Error connecting to server. Make sure backend is running.");
    }
  });
});
