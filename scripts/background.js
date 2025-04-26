// SecureGo Extension - Background Script
// Handles communication between content scripts and the backend API

// API endpoint configuration
const API_CONFIG = {
  baseUrl: "http://localhost:5000", // Default local development server
  endpoints: {
    emailScan: "/AI",
    health: "/health"
  }
};

// Badge colors for different states
const BADGE_COLORS = {
  safe: "#4CAF50",    // Green
  warning: "#FFC107", // Yellow
  danger: "#F44336",  // Red
  default: "#757575"  // Gray
};

// Initialize the extension
function initializeExtension() {
  console.log("SecureGo Extension background script initialized");

  // Check if the API is available
  checkApiHealth();

  // Set default badge
  updateBadge("", BADGE_COLORS.default);
}

// Check the health of the API server
function checkApiHealth() {
  fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.health}`)
    .then(response => response.json())
    .then(data => {
      console.log("API health check:", data);
      if (data.status === "ok" && data.model_loaded) {
        console.log("API is healthy and model is loaded");
      } else {
        console.warn("API is running but model is not loaded");
      }
    })
    .catch(error => {
      console.error("API health check failed:", error);
    });
}

// Update the extension badge
function updateBadge(text, color) {
  chrome.action.setBadgeText({ text: text });
  chrome.action.setBadgeBackgroundColor({ color: color });
}

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("Received message in background script:", message, "from tab:", sender.tab?.id);

  if (message.action === "scanEmail") {
    // Extract email data
    const emailData = message.emailData;

    if (!emailData || !emailData.content) {
      console.error("Invalid email data received:", emailData);
      sendResponse({ status: "error", message: "Invalid email data" });
      return true;
    }

    console.log("Processing email content, length:", emailData.content.length);
    if (emailData.content.length > 100) {
      console.log("Email preview:", emailData.content.substring(0, 100) + "...");
    }

    // Send email data to API for phishing detection
    scanEmail(emailData)
      .then(result => {
        console.log("Phishing scan result:", result);

        // Update badge based on result
        if (result.result === "spam") {
          updateBadge("!", BADGE_COLORS.danger);

          // Show notification for phishing emails
          chrome.notifications.create({
            type: "basic",
            iconUrl: "icons/warning-icon-128.png", // You need to have this icon
            title: "Phishing Alert",
            message: "The email you're viewing might be a phishing attempt. Exercise caution!"
          });
        } else {
          updateBadge("✓", BADGE_COLORS.safe);

          // Update statistics
          updateEmailStatistics(result.result === "spam");
        }

        // Send result back to content script
        console.log("Sending scan results back to content script");
        sendResponse({ status: "success", result: result });
      })
      .catch(error => {
        console.error("Error scanning email:", error);
        updateBadge("?", BADGE_COLORS.warning);
        sendResponse({ status: "error", message: error.message });
      });

    // Return true to indicate we will send a response asynchronously
    return true;
  }
});

// Send email to API for scanning
async function scanEmail(emailData) {
  // Prepare the data for API
  // The API expects a single "email" field with the content
  const payload = {
    email: emailData.content
  };

  console.log(`Sending API request to ${API_CONFIG.baseUrl}${API_CONFIG.endpoints.emailScan}`);

  try {
    // Make API request
    const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.emailScan}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    console.log("API response status:", response.status);

    // Handle API response
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error("API error response:", errorData);
      throw new Error(errorData.error || `API request failed with status: ${response.status}`);
    }

    // Return the scan result
    const result = await response.json();
    console.log("API response data:", result);
    return result;
  } catch (error) {
    console.error("API request failed:", error);
    throw error;
  }
}

// Update email statistics after scans
function updateEmailStatistics(isPhishing) {
  chrome.storage.local.get(['emailStats'], function (result) {
    const stats = result.emailStats || {
      totalScanned: 0,
      phishingDetected: 0,
      lastUpdated: Date.now()
    };

    // Update stats
    stats.totalScanned += 1;
    if (isPhishing) {
      stats.phishingDetected += 1;
    }
    stats.lastUpdated = Date.now();

    // Save updated stats
    chrome.storage.local.set({ emailStats: stats });
  });
}

// Initialize the extension when the script loads
initializeExtension(); 