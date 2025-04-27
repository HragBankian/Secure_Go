// NSFW Detection Script for SecureGo Extension
// Uses nsfwjs library to analyze website content for NSFW material

// Store loaded model
let nsfwModel = null;
let isModelLoading = false;

// Define constants for classification
const NSFW_THRESHOLD = 0.60; // Probability threshold for NSFW content
const NSFW_CLASSES = ['Porn', 'Sexy', 'Hentai']; // NSFW classes from the model

// Domains to exclude from NSFW checking
const NSFW_EXCLUDED_DOMAINS = [
  '10.170.8.90:5000'
];

// Check if a domain should be excluded from NSFW checking
function isExcludedDomain(url) {
  try {
    const urlObj = new URL(url);
    const hostname = urlObj.hostname;
    const hostWithPort = urlObj.host; // includes port if specified
    
    return NSFW_EXCLUDED_DOMAINS.some(excludedDomain => 
      hostname === excludedDomain || 
      hostWithPort === excludedDomain ||
      hostname.endsWith('.' + excludedDomain)
    );
  } catch (e) {
    console.error("Error checking excluded domain:", e);
    return false;
  }
}

// Initialize module
async function initNsfwDetector() {
  console.log("Initializing NSFW detector...");
  
  // Clear any cached results for the excluded domains
  clearExcludedDomainCache();
  
  // Add required script to the page
  injectTensorflowAndNsfwJs();
  
  // Set up click interception for links
  setupLinkInterception();
}

// Clear cached results for excluded domains
function clearExcludedDomainCache() {
  chrome.storage.local.get(['nsfwUrlCache'], (result) => {
    if (!result.nsfwUrlCache) return;
    
    const cache = result.nsfwUrlCache;
    let modified = false;
    
    // Check each cached URL
    for (const url in cache) {
      try {
        const urlObj = new URL(url);
        const hostname = urlObj.hostname;
        const hostWithPort = urlObj.host;
        
        // If URL is from excluded domain, remove from cache
        if (NSFW_EXCLUDED_DOMAINS.some(domain => 
            hostname === domain || 
            hostWithPort === domain || 
            hostname.endsWith('.' + domain))) {
          delete cache[url];
          modified = true;
          console.log(`Removed excluded domain from cache: ${url}`);
        }
      } catch (e) {
        // Skip invalid URLs
      }
    }
    
    // Save updated cache if modified
    if (modified) {
      chrome.storage.local.set({ nsfwUrlCache: cache });
    }
  });
}

// Inject TensorFlow.js and NSFW.js libraries
function injectTensorflowAndNsfwJs() {
  // First inject TensorFlow.js
  const tfScript = document.createElement('script');
  tfScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.11.0/dist/tf.min.js';
  tfScript.onload = () => {
    console.log("TensorFlow.js loaded");
    
    // Then inject NSFW.js
    const nsfwScript = document.createElement('script');
    nsfwScript.src = 'https://cdn.jsdelivr.net/npm/nsfwjs@2.4.2/dist/nsfwjs.min.js';
    nsfwScript.onload = () => {
      console.log("NSFW.js loaded");
    };
    document.head.appendChild(nsfwScript);
  };
  document.head.appendChild(tfScript);
}

// Load the NSFW detection model
async function loadNsfwModel() {
  if (nsfwModel) {
    return nsfwModel; // Return cached model if already loaded
  }
  
  if (isModelLoading) {
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
    isModelLoading = true;
    console.log("Loading NSFW detection model...");
    
    // Send message to background script to show loading indication
    chrome.runtime.sendMessage({
      action: "updateNsfwStatus",
      status: "loading"
    });
    
    // Load the model - use through a message to background script
    // to avoid CORS issues with direct loading
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { action: "loadNsfwModel" },
        (response) => {
          if (response && response.success) {
            nsfwModel = response.model;
            isModelLoading = false;
            console.log("NSFW model loaded successfully");
            resolve(nsfwModel);
          } else {
            isModelLoading = false;
            console.error("Failed to load NSFW model:", response?.error || "Unknown error");
            reject(new Error(response?.error || "Failed to load NSFW model"));
          }
        }
      );
    });
  } catch (error) {
    isModelLoading = false;
    console.error("Error loading NSFW model:", error);
    throw error;
  }
}

// Set up click interception for links
function setupLinkInterception() {
  // Use event delegation for better performance
  document.addEventListener('click', async (event) => {
    // Find if a link was clicked
    let linkElement = event.target;
    while (linkElement && linkElement.tagName !== 'A') {
      linkElement = linkElement.parentElement;
      if (!linkElement || linkElement === document.body) break;
    }
    
    // If a link was clicked with href
    if (linkElement && linkElement.tagName === 'A' && linkElement.href) {
      // Only process http/https links (not javascript: or mailto: etc)
      if (linkElement.href.startsWith('http')) {
        // Prevent the default navigation
        event.preventDefault();
        
        // Check if we should analyze this URL
        chrome.storage.sync.get(['nsfwCheckEnabled'], (result) => {
          const isEnabled = result.nsfwCheckEnabled !== false; // Default to enabled
          
          if (isEnabled) {
            analyzeAndNavigate(linkElement.href, linkElement);
          } else {
            // If disabled, just navigate directly
            window.location.href = linkElement.href;
          }
        });
      }
    }
  }, true); // Use capture phase to intercept before other handlers
}

// Analyze a URL and navigate or show warning based on results
async function analyzeAndNavigate(url, linkElement) {
  // Direct check for specific IP before any other checks
  try {
    const urlObj = new URL(url);
    if (urlObj.host === '10.170.8.90:5000' || urlObj.hostname === '10.170.8.90') {
      console.log("Direct navigation to excluded IP:", urlObj.host);
      window.location.href = url;
      return;
    }
  } catch (e) {
    console.error("Error in direct URL check:", e);
  }

  // Check if the domain is excluded
  if (isExcludedDomain(url)) {
    console.log(`Skipping NSFW check for excluded domain: ${url}`);
    window.location.href = url;
    return;
  }

  try {
    // Show loading indicator
    showLoadingOverlay(linkElement);
    
    // Check if the URL is already in cache
    const cachedResult = await getNsfwCacheResult(url);
    if (cachedResult) {
      // Update stats for using cached result
      updateNsfwStats(url, cachedResult.isNsfw, true);
      
      handleNsfwResult(url, cachedResult, linkElement);
      return;
    }
    
    // Send request to background script to analyze URL
    chrome.runtime.sendMessage(
      {
        action: "analyzeUrlForNsfw",
        url: url
      },
      (response) => {
        // Hide loading indicator
        hideLoadingOverlay();
        
        if (response && response.success) {
          // Cache the result
          setNsfwCacheResult(url, response.result);
          
          // Update stats for fresh analysis
          updateNsfwStats(url, response.result.isNsfw, false);
          
          // Handle the result
          handleNsfwResult(url, response.result, linkElement);
        } else {
          console.error("Failed to analyze URL:", response?.error || "Unknown error");
          // Navigate anyway in case of error
          window.location.href = url;
        }
      }
    );
  } catch (error) {
    hideLoadingOverlay();
    console.error("Error analyzing URL:", error);
    // Navigate anyway in case of error
    window.location.href = url;
  }
}

// Update NSFW statistics when a URL is analyzed
function updateNsfwStats(url, isNsfw, fromCache) {
  chrome.storage.local.get(['nsfwStats', 'nsfwSiteList'], (result) => {
    // Update general stats
    const stats = result.nsfwStats || {
      totalAnalyzed: 0,
      totalNsfw: 0,
      lastUpdated: Date.now()
    };
    
    // For non-cached results, increment the analyzed count
    if (!fromCache) {
      stats.totalAnalyzed++;
      
      // Increment NSFW count if the site is NSFW
      if (isNsfw) {
        stats.totalNsfw++;
      }
    }
    
    stats.lastUpdated = Date.now();
    
    // Update detailed site list
    let siteList = result.nsfwSiteList || [];
    
    // Parse domain from URL
    const domain = extractDomain(url);
    
    // Get current date (YYYY-MM-DD format)
    const today = new Date().toISOString().split('T')[0];
    
    // Find if we already have this domain in our list
    const existingIndex = siteList.findIndex(site => site.domain === domain);
    
    if (existingIndex >= 0) {
      // Update existing entry
      const site = siteList[existingIndex];
      
      // Add new URL if not already in the list of URLs
      if (!site.urls.includes(url)) {
        site.urls.push(url);
      }
      
      // Update visits and date
      site.visits++;
      site.lastVisit = today;
      
      // Update NSFW status if needed
      if (isNsfw && !site.isNsfw) {
        site.isNsfw = true;
        site.nsfwDetectedOn = today;
      }
      
      // Move to top of list (most recent first)
      siteList.splice(existingIndex, 1);
      siteList.unshift(site);
    } else {
      // Add new entry
      siteList.unshift({
        domain: domain,
        urls: [url],
        visits: 1,
        isNsfw: isNsfw,
        firstVisit: today,
        lastVisit: today,
        nsfwDetectedOn: isNsfw ? today : null
      });
    }
    
    // Limit list to most recent 1000 sites
    if (siteList.length > 1000) {
      siteList = siteList.slice(0, 1000);
    }
    
    // Save updated stats and list
    chrome.storage.local.set({
      nsfwStats: stats,
      nsfwSiteList: siteList
    });
    
    // Also update badge with latest status
    chrome.runtime.sendMessage({
      action: "updateNsfwStatus",
      status: isNsfw ? "nsfw" : "safe"
    });
  });
}

// Extract domain from URL
function extractDomain(url) {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname;
  } catch (e) {
    // If URL parsing fails, return original URL
    return url;
  }
}

// Handle NSFW detection result
function handleNsfwResult(url, result, linkElement) {
  if (result.isNsfw) {
    // Show NSFW warning popup
    showNsfwWarning(url, result, () => {
      // Callback when user clicks "Proceed"
      window.location.href = url;
    });
  } else {
    // If safe, navigate directly
    window.location.href = url;
  }
}

// Show NSFW warning popup
function showNsfwWarning(url, result, onProceed) {
  // Force skip warning for specific domains
  try {
    const urlObj = new URL(url);
    if (urlObj.host === '10.170.8.90:5000' || urlObj.hostname === '10.170.8.90') {
      console.log("Forced skipping NSFW warning for excluded IP:", urlObj.host);
      if (onProceed) onProceed(); // Just proceed directly
      return;
    }
  } catch (e) {
    console.error("Error checking URL in showNsfwWarning:", e);
  }

  // Create popup overlay
  const overlay = document.createElement('div');
  overlay.className = 'securego-nsfw-overlay';
  
  // Create popup container
  const popup = document.createElement('div');
  popup.className = 'securego-nsfw-popup';
  
  // Create popup content
  const title = document.createElement('h2');
  title.textContent = 'NSFW Warning';
  
  const message = document.createElement('p');
  message.textContent = `This website may contain content not suitable for all ages (NSFW).`;
  
  const details = document.createElement('div');
  details.className = 'securego-nsfw-details';
  details.innerHTML = `<p>Detection confidence: ${Math.round(result.confidence * 100)}%</p>
    <p>Category: ${result.category}</p>
    <p>Domain: ${extractDomain(url)}</p>`;
  
  // Create buttons
  const buttonContainer = document.createElement('div');
  buttonContainer.className = 'securego-nsfw-buttons';
  
  const cancelButton = document.createElement('button');
  cancelButton.textContent = 'Go Back';
  cancelButton.className = 'securego-nsfw-btn-cancel';
  cancelButton.onclick = () => {
    document.body.removeChild(overlay);
  };
  
  const proceedButton = document.createElement('button');
  proceedButton.textContent = 'Proceed Anyway';
  proceedButton.className = 'securego-nsfw-btn-proceed';
  proceedButton.onclick = () => {
    // Record user's decision to proceed to NSFW site
    recordUserAction(url, 'proceed');
    
    document.body.removeChild(overlay);
    if (onProceed) onProceed();
  };
  
  // Assemble popup
  buttonContainer.appendChild(cancelButton);
  buttonContainer.appendChild(proceedButton);
  
  popup.appendChild(title);
  popup.appendChild(message);
  popup.appendChild(details);
  popup.appendChild(buttonContainer);
  
  overlay.appendChild(popup);
  
  // Add styles
  addNsfwStyles();
  
  // Add to page
  document.body.appendChild(overlay);
  
  // Record this warning was shown
  recordUserAction(url, 'warning_shown');
}

// Record user action for analytics
function recordUserAction(url, action) {
  chrome.storage.local.get(['nsfwUserActions'], (result) => {
    const actions = result.nsfwUserActions || [];
    
    // Add new action to the beginning
    actions.unshift({
      url: url,
      domain: extractDomain(url),
      action: action,
      timestamp: Date.now(),
      date: new Date().toISOString()
    });
    
    // Keep only the most recent 500 actions
    if (actions.length > 500) {
      actions.length = 500;
    }
    
    chrome.storage.local.set({ nsfwUserActions: actions });
  });
}

// Add styles for NSFW warning popup
function addNsfwStyles() {
  if (document.getElementById('securego-nsfw-styles')) return;
  
  const styleElement = document.createElement('style');
  styleElement.id = 'securego-nsfw-styles';
  styleElement.textContent = `
    .securego-nsfw-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.7);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 999999;
      backdrop-filter: blur(5px);
    }
    
    .securego-nsfw-popup {
      background-color: white;
      border-radius: 8px;
      padding: 24px;
      max-width: 450px;
      width: 80%;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .securego-nsfw-popup h2 {
      color: #d32f2f;
      margin-top: 0;
      font-size: 24px;
    }
    
    .securego-nsfw-popup p {
      font-size: 16px;
      line-height: 1.5;
      color: #333;
    }
    
    .securego-nsfw-details {
      background-color: #f5f5f5;
      padding: 12px;
      border-radius: 4px;
      margin: 16px 0;
    }
    
    .securego-nsfw-details p {
      margin: 8px 0;
      font-size: 14px;
      color: #555;
    }
    
    .securego-nsfw-buttons {
      display: flex;
      justify-content: flex-end;
      gap: 12px;
      margin-top: 16px;
    }
    
    .securego-nsfw-btn-cancel,
    .securego-nsfw-btn-proceed {
      padding: 10px 16px;
      border-radius: 4px;
      font-weight: 500;
      cursor: pointer;
      border: none;
    }
    
    .securego-nsfw-btn-cancel {
      background-color: #f5f5f5;
      color: #333;
    }
    
    .securego-nsfw-btn-proceed {
      background-color: #d32f2f;
      color: white;
    }
    
    .securego-nsfw-loading {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(255, 255, 255, 0.8);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 10;
      border-radius: 4px;
    }
    
    .securego-nsfw-spinner {
      width: 24px;
      height: 24px;
      border: 3px solid #f3f3f3;
      border-top: 3px solid #3498db;
      border-radius: 50%;
      animation: securego-nsfw-spin 1s linear infinite;
    }
    
    @keyframes securego-nsfw-spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
  
  document.head.appendChild(styleElement);
}

// Show loading overlay on link
function showLoadingOverlay(linkElement) {
  // Don't show loading for non-element
  if (!linkElement || !linkElement.getBoundingClientRect) return;
  
  // Create loading overlay
  const overlay = document.createElement('div');
  overlay.className = 'securego-nsfw-loading';
  overlay.id = 'securego-nsfw-loading';
  
  // Create spinner
  const spinner = document.createElement('div');
  spinner.className = 'securego-nsfw-spinner';
  overlay.appendChild(spinner);
  
  // Position the overlay over the link
  const rect = linkElement.getBoundingClientRect();
  overlay.style.position = 'fixed';
  overlay.style.top = `${rect.top}px`;
  overlay.style.left = `${rect.left}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  
  // Add styles if not already added
  addNsfwStyles();
  
  // Add to page
  document.body.appendChild(overlay);
}

// Hide loading overlay
function hideLoadingOverlay() {
  const overlay = document.getElementById('securego-nsfw-loading');
  if (overlay) {
    document.body.removeChild(overlay);
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

// Initialize on page load
initNsfwDetector(); 