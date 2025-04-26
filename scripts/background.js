// SecureGo Extension - Background Script
// Handles communication between content scripts and the backend API

// API endpoint configuration
const API_CONFIG = {
  baseUrl: "http://localhost:5000", // Default local development server
  endpoints: {
    emailScan: "/api/scan/email",
    health: "/health",
    urlScan: "/api/scan/url",
    urlBatchScan: "/api/scan/urls" // New batch endpoint
  }
};

// Badge colors for different states
const BADGE_COLORS = {
  safe: "#4CAF50",    // Green
  warning: "#FFC107", // Yellow
  danger: "#F44336",  // Red
  default: "#757575"  // Gray
};

// Extension state
let API_AVAILABLE = false;
let API_CHECK_INTERVAL = null;
let RETRY_INTERVAL = 5000; // 5 seconds between retries
const MAX_BATCH_SIZE = 50; // Maximum number of URLs to process in a single batch
const REQUEST_TIMEOUT = 10000; // 10 second timeout for API requests
const MAX_CONCURRENT_REQUESTS = 3; // Maximum number of concurrent API requests
let activeRequestCount = 0; // Track current number of active requests

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
  console.log('Checking API health...');

  // Request with timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.health}`, {
    method: 'GET',
    signal: controller.signal
  })
    .then(response => {
      clearTimeout(timeoutId);
      if (response.ok) {
        return response.json();
      }
      throw new Error(`API returned status ${response.status}`);
    })
    .then(data => {
      console.log('API health check response:', data);
      if (data.status === 'healthy' || data.status === 'degraded') {
        console.log('API is available');
        API_AVAILABLE = true;
        updateBadge(true);

        // If we have a regular check interval, clear it and set a longer one
        if (API_CHECK_INTERVAL) {
          clearInterval(API_CHECK_INTERVAL);
          API_CHECK_INTERVAL = setInterval(checkApiHealth, 60000); // Check every minute when healthy
        }
      } else {
        console.warn('API reported unhealthy status:', data);
        API_AVAILABLE = false;
        updateBadge(false);

        // Set shorter retry interval
        if (API_CHECK_INTERVAL) {
          clearInterval(API_CHECK_INTERVAL);
        }
        API_CHECK_INTERVAL = setInterval(checkApiHealth, RETRY_INTERVAL);
      }
    })
    .catch(error => {
      clearTimeout(timeoutId);
      console.error('Error checking API health:', error);
      API_AVAILABLE = false;
      updateBadge(false);

      // Try again sooner if API is down
      if (API_CHECK_INTERVAL) {
        clearInterval(API_CHECK_INTERVAL);
      }
      API_CHECK_INTERVAL = setInterval(checkApiHealth, RETRY_INTERVAL);
    });
}

// Update the extension badge
function updateBadge(isAvailable) {
  chrome.action.setBadgeText({
    text: isAvailable ? 'ON' : 'OFF'
  });
  chrome.action.setBadgeBackgroundColor({
    color: isAvailable ? BADGE_COLORS.safe : BADGE_COLORS.danger
  });
}

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("Received message in background script:", message);

  // Handle URL scanning requests
  if (message.action === "scanUrls") {
    scanUrls(message.urlData, sender.tab.id, sendResponse);
    return true; // Return true to indicate we will respond asynchronously
  }

  // Handle email scanning requests
  if (message.action === "scanEmail") {
    scanEmail(message.emailData, sender.tab.id, sendResponse);
    return true; // Return true to indicate we will respond asynchronously
  }

  // Handle batch URL scanning requests
  if (message.action === "scanUrlsBatch") {
    scanUrlsBatch(message.urlData, sender.tab.id, sendResponse);
    return true; // Return true to indicate we will respond asynchronously
  }

  // Handle settings retrieval
  if (message.action === "getSettings") {
    // Retrieve settings and send them back
    chrome.storage.sync.get(['urlScannerEnabled', 'emailScannerEnabled'], function (result) {
      sendResponse({
        success: true,
        settings: result
      });
    });
    return true; // Return true to indicate we will respond asynchronously
  }

  // Handle API health check
  if (message.action === "checkApiHealth") {
    // Return current API status and trigger a fresh check
    setTimeout(checkApiHealth, 100);
    sendResponse({ available: API_AVAILABLE });
    return false;
  }
});

// Queue manager for throttling API requests
const requestQueue = [];
const REQUEST_QUEUE_CHECK_INTERVAL = 100; // Check queue every 100ms

// Start queue processing
setInterval(processRequestQueue, REQUEST_QUEUE_CHECK_INTERVAL);

// Process requests from the queue
function processRequestQueue() {
  // If we can't process more requests or queue is empty, do nothing
  if (activeRequestCount >= MAX_CONCURRENT_REQUESTS || requestQueue.length === 0) {
    return;
  }

  // Get the next request
  const nextRequest = requestQueue.shift();

  // Execute the request
  nextRequest.execute();
}

// Add a request to the queue
function queueRequest(requestFunction) {
  requestQueue.push({ execute: requestFunction });
  console.log(`Request queued. Queue length: ${requestQueue.length}`);
}

// Execute an API request with tracking and timeouts
function executeApiRequest(url, options, onSuccess, onError) {
  // Increment active request counter
  activeRequestCount++;

  // Add timeout to request
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  // Add signal to options
  options.signal = controller.signal;

  console.log(`Executing API request to ${url}`);

  fetch(url, options)
    .then(response => {
      clearTimeout(timeoutId);
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log(`API response from ${url}:`, data);
      onSuccess(data);
    })
    .catch(error => {
      clearTimeout(timeoutId);
      console.error(`Error in API request to ${url}:`, error);
      onError(error);
    })
    .finally(() => {
      // Decrement active request counter
      activeRequestCount--;
    });
}

// Function to scan URLs through the API
function scanUrls(urlData, tabId, sendResponse) {
  console.log(`Scanning ${urlData.links.length} URLs for tab ${tabId}`);

  if (!API_AVAILABLE) {
    console.warn("API is not available. Cannot scan URLs.");
    sendResponse({
      success: false,
      error: "API is not available"
    });
    return;
  }

  // Queue the request
  queueRequest(() => {
    executeApiRequest(
      `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.urlScan}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          url: urlData.links[0].url
        })
      },
      (data) => {
        console.log("URL scan result:", data);
        sendResponse({
          success: true,
          results: [{
            url: data.url,
            isMalicious: data.isMalicious
          }]
        });
      },
      (error) => {
        sendResponse({
          success: false,
          error: error.message
        });
      }
    );
  });
}

// Function to scan URLs in batch through the API
function scanUrlsBatch(urlData, tabId, sendResponse) {
  console.log(`Batch scanning ${urlData.urls.length} URLs for tab ${tabId}`);

  if (!API_AVAILABLE) {
    console.warn("API is not available. Cannot scan URLs in batch.");
    sendResponse({
      success: false,
      error: "API is not available"
    });
    return;
  }

  // Queue the request
  queueRequest(() => {
    executeApiRequest(
      `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.urlBatchScan}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          urls: urlData.urls
        })
      },
      (data) => {
        console.log("Batch URL scan results:", data);

        // Extract results from the API response
        if (data.results && Array.isArray(data.results)) {
          // Map the API response to a format expected by the content script
          const results = data.results.map(item => ({
            url: item.url,
            isMalicious: item.isMalicious
          }));

          console.log("Processed results for content script:", results);

          // Send the results back to the content script
          sendResponse({
            success: true,
            results: results
          });
        } else {
          console.error("Unexpected API response format:", data);
          sendResponse({
            success: false,
            error: "Invalid API response format"
          });
        }
      },
      (error) => {
        console.error("Error batch scanning URLs:", error);
        sendResponse({
          success: false,
          error: error.message
        });
      }
    );
  });
}

// Function to scan emails through the API
function scanEmail(emailData, tabId, sendResponse) {
  console.log(`Scanning email for tab ${tabId}`);

  if (!API_AVAILABLE) {
    console.warn("API is not available. Cannot scan email.");
    sendResponse({
      success: false,
      error: "API is not available"
    });
    return;
  }

  // Queue the request
  queueRequest(() => {
    executeApiRequest(
      `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.emailScan}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: emailData.content,
          subject: emailData.subject || "",
          sender: emailData.sender || ""
        })
      },
      (data) => {
        console.log("Email scan results:", data);

        // Map the API response to the expected format
        sendResponse({
          success: data.success,
          isPhishing: data.result === 'spam',
          result: data.result,
          originalLabel: data.original_label,
          confidence: data.confidence || 0.9,
          explanation: "Analysis complete.",
          emailPreview: data.email_preview
        });
      },
      (error) => {
        console.error("Error scanning email:", error);
        sendResponse({
          success: false,
          error: error.message
        });
      }
    );
  });
}

// Extension installation or update listener
chrome.runtime.onInstalled.addListener(function (details) {
  if (details.reason === "install") {
    // Initialize default settings
    chrome.storage.sync.set({
      urlScannerEnabled: true,
      emailScannerEnabled: true
    }, function () {
      console.log("Default settings initialized");
    });

    // Open onboarding page
    chrome.tabs.create({
      url: chrome.runtime.getURL("html/onboarding.html")
    });
  } else if (details.reason === "update") {
    console.log("Extension updated to version", chrome.runtime.getManifest().version);
  }
});

// Initialize the extension when the script loads
initializeExtension(); 