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

// NSFW Detection state
let nsfwModel = null;
let isNsfwModelLoading = false;
const NSFW_MODEL_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/nsfwjs/model.json';
const NSFW_THRESHOLD = 0.60; // Probability threshold for NSFW content

// Initialize the extension
function initializeExtension() {
  console.log("SecureGo Extension background script initialized");

  // Check if the API is available
  checkApiHealth();

  // Set default badge
  updateBadge("", BADGE_COLORS.default);
  
  // Listen for tab updates to reset the badge
  chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete') {
      updateBadge("", BADGE_COLORS.default);
    }
  });
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
      if (data.status === 'healthy' || data.status === 'ok' || data.status === 'degraded') {
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

  // Handle NSFW model loading
  if (message.action === "loadNsfwModel") {
    loadNsfwModel()
      .then(model => {
        sendResponse({
          success: true,
          model: model
        });
      })
      .catch(error => {
        sendResponse({
          success: false,
          error: error.message
        });
      });
    return true; // Return true to indicate we will respond asynchronously
  }
  
  // Handle URL analysis for NSFW content
  if (message.action === "analyzeUrlForNsfw") {
    analyzeUrlForNsfw(message.url)
      .then(result => {
        sendResponse({
          success: true,
          result: result
        });
      })
      .catch(error => {
        sendResponse({
          success: false,
          error: error.message
        });
      });
    return true; // Return true to indicate we will respond asynchronously
  }
  
  // Handle NSFW status updates
  if (message.action === "updateNsfwStatus") {
    updateNsfwStatus(message.status, sender.tab.id);
    sendResponse({ success: true });
    return false;
  }

  // Handle settings retrieval
  if (message.action === "getSettings") {
    // Retrieve settings and send them back
    chrome.storage.sync.get(
      ['urlScannerEnabled', 'emailScannerEnabled', 'nsfwCheckEnabled'], 
      function (result) {
        sendResponse({
          success: true,
          settings: result
        });
      }
    );
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

// NSFW detection functions

// Load the NSFW detection model
async function loadNsfwModel() {
  if (nsfwModel) {
    return nsfwModel; // Return cached model if already loaded
  }
  
  if (isNsfwModelLoading) {
    // Wait for the model to finish loading
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (nsfwModel) {
          clearInterval(checkInterval);
          resolve(nsfwModel);
        }
      }, 100);
    });
  }
  
  try {
    isNsfwModelLoading = true;
    console.log("Loading NSFW detection model from:", NSFW_MODEL_URL);
    
    // Import TensorFlow.js and NSFWJS dynamically
    // In background script, we need to use importScripts
    // However, for security reasons this won't work with the CDN directly
    // We'll use a simple proxy approach to fetch the image and analyze it
    
    // Since direct TF.js use is complex in extension background scripts,
    // we'll use a different approach: taking screenshots and analyzing them
    
    // For demo/MVP purposes, we'll use a simpler approach that doesn't
    // require actual TF.js model loading in the background
    
    isNsfwModelLoading = false;
    nsfwModel = {
      loaded: true,
      timestamp: Date.now()
    };
    
    console.log("NSFW detection initialized");
    return nsfwModel;
  } catch (error) {
    isNsfwModelLoading = false;
    console.error("Error loading NSFW model:", error);
    throw error;
  }
}

// Analyze a URL for NSFW content
async function analyzeUrlForNsfw(url) {
  try {
    console.log("Analyzing URL for NSFW content:", url);
    
    // First check in cache
    const cachedResult = await getNsfwCacheResult(url);
    if (cachedResult) {
      console.log("Using cached NSFW result for:", url);
      return cachedResult;
    }
    
    // In a real implementation, we would:
    // 1. Take a screenshot of the page
    // 2. Use the NSFW model to analyze the image
    // 3. Return the result
    
    // For demonstration purposes, we'll use domain-based classification
    // This is NOT a real NSFW detection approach but simulates the behavior
    const domainInfo = new URL(url);
    const domain = domainInfo.hostname.toLowerCase();
    
    // Demo: Classify certain domains as NSFW for testing
    const nsfwTestDomains = ['adult', 'xxx', 'porn', 'sex', 'nsfw'];
    
    // Check if domain contains any NSFW terms
    const isLikelyNsfw = nsfwTestDomains.some(term => domain.includes(term));
    
    // Demo classification
    let result;
    if (isLikelyNsfw) {
      result = {
        isNsfw: true,
        confidence: 0.95,
        category: 'Adult Content',
        timestamp: Date.now()
      };
    } else {
      // Random small chance of false positive for testing
      const randomValue = Math.random();
      if (randomValue < 0.05) { // 5% chance
        result = {
          isNsfw: true,
          confidence: 0.65 + (randomValue / 5), // Between 0.65 and 0.85
          category: 'Possible Adult Content',
          timestamp: Date.now()
        };
      } else {
        result = {
          isNsfw: false,
          confidence: 0.9,
          category: 'Safe Content',
          timestamp: Date.now()
        };
      }
    }
    
    // Cache the result
    await setNsfwCacheResult(url, result);
    
    console.log("NSFW analysis result for", url, ":", result);
    return result;
  } catch (error) {
    console.error("Error analyzing URL for NSFW content:", error);
    
    // In case of error, default to letting the user proceed
    return {
      isNsfw: false,
      confidence: 0,
      category: 'Error - Could not analyze',
      error: error.message,
      timestamp: Date.now()
    };
  }
}

// Update badge to show NSFW status
function updateNsfwStatus(status, tabId) {
  if (status === 'loading') {
    chrome.action.setBadgeText({
      text: '...',
      tabId: tabId
    });
    chrome.action.setBadgeBackgroundColor({
      color: BADGE_COLORS.warning,
      tabId: tabId
    });
  } else if (status === 'nsfw') {
    chrome.action.setBadgeText({
      text: '18+',
      tabId: tabId
    });
    chrome.action.setBadgeBackgroundColor({
      color: BADGE_COLORS.danger,
      tabId: tabId
    });
  } else if (status === 'safe') {
    chrome.action.setBadgeText({
      text: 'OK',
      tabId: tabId
    });
    chrome.action.setBadgeBackgroundColor({
      color: BADGE_COLORS.safe,
      tabId: tabId
    });
  }
}

// Cache functions for NSFW results
async function getNsfwCacheResult(url) {
  return new Promise((resolve) => {
    chrome.storage.local.get(['nsfwUrlCache'], (result) => {
      const cache = result.nsfwUrlCache || {};
      const cachedResult = cache[url];
      
      // Return cached result if it exists and is not too old (1 day)
      if (cachedResult && (Date.now() - cachedResult.timestamp < 86400000)) {
        resolve(cachedResult);
      } else {
        resolve(null);
      }
    });
  });
}

async function setNsfwCacheResult(url, result) {
  return new Promise((resolve) => {
    chrome.storage.local.get(['nsfwUrlCache'], (data) => {
      const cache = data.nsfwUrlCache || {};
      
      // Add result to cache with timestamp
      cache[url] = {
        ...result,
        timestamp: Date.now()
      };
      
      // Enforce max cache size (100 items)
      const urls = Object.keys(cache);
      if (urls.length > 100) {
        // Remove oldest items
        urls.sort((a, b) => cache[a].timestamp - cache[b].timestamp);
        urls.slice(0, urls.length - 100).forEach(oldUrl => {
          delete cache[oldUrl];
        });
      }
      
      // Save updated cache
      chrome.storage.local.set({ nsfwUrlCache: cache }, resolve);
    });
  });
}

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
      console.log(`API request to ${url} succeeded:`, data);
      activeRequestCount--;
      onSuccess(data);
    })
    .catch(error => {
      clearTimeout(timeoutId);
      console.error(`API request to ${url} failed:`, error);
      activeRequestCount--;
      onError(error);
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

        // Debug the returned data structure
        console.log("API response structure:", Object.keys(data));
        console.log("API response 'result' field:", data.result);
        console.log("API response 'success' field:", data.success);

        // Map the API response to the expected format
        // Ensure we include all necessary fields for the content script
        sendResponse({
          success: data.success === true,
          isPhishing: data.result === 'spam',
          result: data.result || 'unknown',
          originalLabel: data.original_label || data.result,
          confidence: data.confidence || 0.9,
          explanation: "Analysis complete.",
          emailPreview: data.email_preview || emailData.content.substring(0, 100)
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