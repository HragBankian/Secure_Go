// Background script for SecureGo extension
// Handles communication with backend services

// Configuration for API endpoints
const API_ENDPOINTS = {
  emailAnalysis: "https://api.securego.example/analyze-email", // Replace with your actual API endpoint
  urlScanning: "https://api.securego.example/scan-urls" // Replace with your actual API endpoint
};

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Handle email scanning requests
  if (message.action === "scanEmail") {
    // Log the received email data (for development purposes)
    console.log("Received email data:", message.emailData);
    
    // Forward the email data to the backend
    sendEmailToBackend(message.emailData)
      .then(response => {
        console.log("Email data sent to backend successfully");
        sendResponse({ success: true });
      })
      .catch(error => {
        console.error("Error sending email data to backend:", error);
        sendResponse({ success: false, error: error.message });
      });
    
    // Return true to indicate we'll respond asynchronously
    return true;
  }
  
  // Handle URL scanning requests
  if (message.action === "scanUrls") {
    // Log the received URL data (for development purposes)
    console.log("Received URL data for scanning:", message.urlData);
    
    // Check if URL scanning is enabled
    chrome.storage.sync.get(['urlScannerEnabled'], function(result) {
      if (result.urlScannerEnabled !== false) { // Default to enabled if not set
        // Send URLs to backend for scanning
        sendUrlsToBackend(message.urlData)
          .then(results => {
            console.log("URL scan results received from backend:", results);
            sendResponse({ success: true, results: results });
          })
          .catch(error => {
            console.error("Error scanning URLs:", error);
            sendResponse({ success: false, error: error.message });
          });
      } else {
        console.log("URL scanning is disabled");
        sendResponse({ success: false, disabled: true });
      }
    });
    
    // Return true to indicate we'll respond asynchronously
    return true;
  }
});

/**
 * Send email data to the backend for analysis
 * @param {Object} emailData - Object containing email information
 * @returns {Promise} - Promise resolving with the API response
 */
async function sendEmailToBackend(emailData) {
  try {
    // Prepare the request payload with the raw email data
    const payload = {
      sender: emailData.sender,
      subject: emailData.subject,
      content: emailData.content,
      timestamp: new Date().toISOString(),
      source: "chrome-extension"
    };
    
    console.log("Sending email payload to backend:", payload);
    
    // Send the POST request to the backend
    const response = await fetch(API_ENDPOINTS.emailAnalysis, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    // Handle the response
    if (!response.ok) {
      throw new Error(`API request failed with status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error in sendEmailToBackend:", error);
    throw error;
  }
}

/**
 * Send URLs to the backend for scanning
 * @param {Object} urlData - Object containing URLs to scan
 * @returns {Promise} - Promise resolving with scan results
 */
async function sendUrlsToBackend(urlData) {
  try {
    // Prepare the request payload with URL data
    const payload = {
      pageUrl: urlData.pageUrl,
      links: urlData.links,
      timestamp: new Date().toISOString(),
      source: "chrome-extension"
    };
    
    console.log("Sending URL payload to backend:", payload);
    
    // Send the POST request to the backend
    const response = await fetch(API_ENDPOINTS.urlScanning, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    
    // Handle the response
    if (!response.ok) {
      throw new Error(`URL scanning API request failed with status: ${response.status}`);
    }
    
    const data = await response.json();
    
    // Expected response format:
    // { results: [{ url: "https://example.com", isMalicious: true }, ...] }
    return data.results || [];
  } catch (error) {
    console.error("Error in sendUrlsToBackend:", error);
    throw error;
  }
} 