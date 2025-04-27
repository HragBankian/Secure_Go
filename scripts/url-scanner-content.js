// URL Scanner Content Script
// This script examines links on web pages and highlights potentially malicious URLs

// Disable the URL scanner initialization
const URL_SCANNER_ENABLED = false;

// Domains to exclude from URL scanning
const EXCLUDED_DOMAINS = [
  '10.170.8.90:5000'
];

// Specific IP check - more direct than the domain exclusion
function isSpecificIp(url) {
  try {
    const urlObj = new URL(url);
    return urlObj.host === '10.170.8.90:5000' || urlObj.hostname === '10.170.8.90';
  } catch (e) {
    return false;
  }
}

// Cache of previously scanned URLs to avoid rescanning
const urlScanCache = new Map();

// Set of URLs that have been processed to avoid duplicate scanning
const processedUrls = new Set();

// Queue for URLs waiting to be scanned in a batch
let urlScanQueue = [];
// Priority queue for visible URLs
let visibleUrlQueue = [];

// Throttling and batch configuration
const SCAN_DEBOUNCE_TIME = 500; // 0.5 second (reduced from 1s)
const MAX_BATCH_SIZE = 50; // Increased from 20 for better throughput
const MIN_BATCH_SIZE = 10; // Minimum size to trigger a batch
const MAX_CONCURRENT_BATCHES = 2; // Maximum number of concurrent batch requests
let activeBatchCount = 0; // Current number of active batch scan requests
let scanDebounceTimer = null;

// Exponential backoff for retries
const MAX_RETRY_ATTEMPTS = 3;
const RETRY_DELAY_BASE = 1000; // Base delay in ms
let retryAttempts = 0;

// Debug mode - set to true to enable verbose logging
const DEBUG_MODE = true;

// Initialize URL scanner
function initializeUrlScanner() {
  console.log("URL scanner disabled by configuration.");
  
  // Check if the scanner is enabled
  if (!URL_SCANNER_ENABLED) {
    console.log("URL scanner is disabled.");
    return;
  }

  console.log("Initializing URL scanner...");

  // Set up intersection observer to detect visible links
  setupVisibilityObserver();

  // Extract and scan all URLs on the page
  extractAndScanUrls();

  // Set up observer to detect new links being added to the page
  const observer = new MutationObserver(handleMutations);

  // Start observing page changes
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

  console.log("URL scanner initialized and watching for new links");
}

// Debugging helper function
function debug(...args) {
  if (DEBUG_MODE) {
    console.log("[SecureGo Debug]", ...args);
  }
}

// Set up intersection observer to prioritize visible links
function setupVisibilityObserver() {
  // Create IntersectionObserver to detect visible links
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && entry.target.tagName === 'A') {
        const link = entry.target;
        if (isValidUrl(link.href)) {
          const normalizedUrl = normalizeUrl(link.href);

          // If not already processed and not in cache, prioritize it
          if (!processedUrls.has(normalizedUrl) && !urlScanCache.has(normalizedUrl)) {
            processedUrls.add(normalizedUrl);
            prioritizeUrlForScanning(normalizedUrl, link);
          } else if (urlScanCache.has(normalizedUrl)) {
            // Apply cached result
            applyResultToLink(link, urlScanCache.get(normalizedUrl));
          }
        }
      }
    });
  }, {
    threshold: 0.1, // 10% visibility is enough to trigger
    rootMargin: '200px' // Check links that are close to viewport
  });

  // Observe all links
  document.querySelectorAll('a[href]').forEach(link => {
    observer.observe(link);
  });

  // Store observer reference for later use with new links
  window.secureGoVisibilityObserver = observer;
}

// Handle DOM mutations (new links added)
function handleMutations(mutations) {
  let hasNewLinks = false;

  mutations.forEach(mutation => {
    // Check if any added nodes contain links
    if (mutation.addedNodes && mutation.addedNodes.length > 0) {
      for (let i = 0; i < mutation.addedNodes.length; i++) {
        const node = mutation.addedNodes[i];

        // Check if the node is an element node with a tag name (not a text node)
        if (node.nodeType === 1) {
          // If the node itself is a link
          if (node.tagName === 'A' && isValidUrl(node.href)) {
            hasNewLinks = true;
            processSingleLink(node);

            // Observe this link for visibility
            if (window.secureGoVisibilityObserver) {
              window.secureGoVisibilityObserver.observe(node);
            }
          }
          // Or if it contains links
          else if (node.querySelectorAll) {
            const links = node.querySelectorAll('a[href]');
            if (links.length > 0) {
              hasNewLinks = true;

              // Process all new links found
              links.forEach(link => {
                processSingleLink(link);

                // Observe this link for visibility
                if (window.secureGoVisibilityObserver) {
                  window.secureGoVisibilityObserver.observe(link);
                }
              });
            }
          }
        }
      }
    }
  });

  // If new links were found and either queue has content, initiate scanning
  if (hasNewLinks && (visibleUrlQueue.length > 0 || urlScanQueue.length > 0)) {
    // If there are enough URLs in the visible queue or the main queue, process immediately
    if (visibleUrlQueue.length >= MIN_BATCH_SIZE || urlScanQueue.length >= MAX_BATCH_SIZE) {
      if (scanDebounceTimer) {
        clearTimeout(scanDebounceTimer);
      }
      processBatchScan();
    } else {
      // Otherwise debounce
      if (scanDebounceTimer) {
        clearTimeout(scanDebounceTimer);
      }
      scanDebounceTimer = setTimeout(() => {
        processBatchScan();
      }, SCAN_DEBOUNCE_TIME);
    }
  }
}

// Process a single link
function processSingleLink(link) {
  // Skip specific IPs immediately
  if (link && link.href && isSpecificIp(link.href)) {
    return;
  }
  
  if (isValidUrl(link.href)) {
    const normalizedUrl = normalizeUrl(link.href);
    if (!processedUrls.has(normalizedUrl)) {
      processedUrls.add(normalizedUrl);

      // If we already have the result cached, apply it immediately
      if (urlScanCache.has(normalizedUrl)) {
        applyResultToLink(link, urlScanCache.get(normalizedUrl));
      } else {
        // Check if link is in viewport
        const rect = link.getBoundingClientRect();
        const isVisible = (
          rect.top >= 0 &&
          rect.left >= 0 &&
          rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
          rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );

        // Prioritize visible links
        if (isVisible) {
          prioritizeUrlForScanning(normalizedUrl, link);
        } else {
          queueUrlForScanning(normalizedUrl);
        }
      }
    }
  }
}

// More efficient URL normalization
function normalizeUrl(url) {
  // Common prefixes to strip (avoid creating URL objects which is expensive)
  const commonPrefixes = ['https://www.', 'http://www.', 'https://', 'http://'];

  try {
    // Remove fragment part (everything after #) - quick string operation
    let cleanUrl = url.split('#')[0];

    // Only create a URL object if necessary for complex URLs
    if (cleanUrl.includes('?') || !commonPrefixes.some(prefix => cleanUrl.startsWith(prefix))) {
      const urlObj = new URL(cleanUrl);
      return `${urlObj.protocol}//${urlObj.hostname}${urlObj.pathname}`;
    } else {
      // For simple URLs, manual string manipulation is faster
      for (const prefix of commonPrefixes) {
        if (cleanUrl.startsWith(prefix)) {
          cleanUrl = cleanUrl.substring(prefix.length);
          break;
        }
      }

      // Remove trailing slash if present
      if (cleanUrl.endsWith('/')) {
        cleanUrl = cleanUrl.slice(0, -1);
      }

      // Get domain and path
      const pathIndex = cleanUrl.indexOf('/');
      if (pathIndex > 0) {
        return cleanUrl;
      } else {
        return cleanUrl;
      }
    }
  } catch (error) {
    // If URL parsing fails, return original URL
    console.warn("Failed to normalize URL:", url);
    return url;
  }
}

// Extract and scan all URLs on the page
function extractAndScanUrls() {
  console.log("Extracting URLs from page...");

  // Get all links on the page
  const links = document.querySelectorAll('a[href]');
  const validLinks = [];
  let visibleLinksCount = 0;

  console.log(`Found ${links.length} links on the page`);

  // Check what links are visible in the viewport
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
  const viewportWidth = window.innerWidth || document.documentElement.clientWidth;

  // Filter for valid URLs and process them
  links.forEach(link => {
    if (isValidUrl(link.href)) {
      const normalizedUrl = normalizeUrl(link.href);

      // Skip URLs we've already processed
      if (!processedUrls.has(normalizedUrl)) {
        processedUrls.add(normalizedUrl);

        // If we already have the result cached, apply it immediately
        if (urlScanCache.has(normalizedUrl)) {
          applyResultToLink(link, urlScanCache.get(normalizedUrl));
        } else {
          // Check if the link is visible in the viewport
          const rect = link.getBoundingClientRect();
          const isVisible = (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= viewportHeight &&
            rect.right <= viewportWidth
          );

          // Prioritize visible links
          if (isVisible) {
            visibleLinksCount++;
            prioritizeUrlForScanning(normalizedUrl, link);
          } else {
            validLinks.push({ link, url: normalizedUrl });
            queueUrlForScanning(normalizedUrl);
          }
        }
      }
    }
  });

  console.log(`Prioritized ${visibleLinksCount} visible links and queued ${urlScanQueue.length} for scanning`);

  // If we have enough visible links, process them immediately
  if (visibleUrlQueue.length > 0) {
    processBatchScan();
  }
}

// Prioritize visible URLs for immediate scanning
function prioritizeUrlForScanning(url, linkElement) {
  visibleUrlQueue.push({ url, linkElement });

  // Clear any existing timer
  if (scanDebounceTimer) {
    clearTimeout(scanDebounceTimer);
  }

  // If we have enough for a batch, process immediately
  if (visibleUrlQueue.length >= MIN_BATCH_SIZE) {
    processBatchScan();
  } else {
    // Otherwise set a short timer
    scanDebounceTimer = setTimeout(() => {
      processBatchScan();
    }, 200); // Faster debounce time for visible links
  }
}

// Queue a URL for scanning
function queueUrlForScanning(url) {
  // Add URL to the queue
  urlScanQueue.push(url);

  // Clear any existing timer
  if (scanDebounceTimer) {
    clearTimeout(scanDebounceTimer);
  }

  // Set a new timer to process the batch
  scanDebounceTimer = setTimeout(() => {
    processBatchScan();
  }, SCAN_DEBOUNCE_TIME);
}

// Process batch of URLs for scanning
function processBatchScan() {
  // If we already have too many concurrent batches, just return and wait
  if (activeBatchCount >= MAX_CONCURRENT_BATCHES) {
    // But set a timer to check again later
    scanDebounceTimer = setTimeout(() => {
      processBatchScan();
    }, 100);
    return;
  }

  // If both queues are empty, do nothing
  if (visibleUrlQueue.length === 0 && urlScanQueue.length === 0) {
    return;
  }

  // Prioritize visible links first, then regular queue
  let currentBatch = [];
  let linkElements = new Map();

  // First take from the priority queue (visible links)
  if (visibleUrlQueue.length > 0) {
    const visibleItems = visibleUrlQueue.splice(0, Math.min(MAX_BATCH_SIZE, visibleUrlQueue.length));
    visibleItems.forEach(item => {
      currentBatch.push(item.url);
      linkElements.set(item.url, item.linkElement);
    });
  }

  // If we need more items, take from the regular queue
  if (currentBatch.length < MAX_BATCH_SIZE && urlScanQueue.length > 0) {
    const remaining = MAX_BATCH_SIZE - currentBatch.length;
    const regularItems = urlScanQueue.splice(0, Math.min(remaining, urlScanQueue.length));
    currentBatch = currentBatch.concat(regularItems);
  }

  console.log(`Processing batch of ${currentBatch.length} URLs (${activeBatchCount + 1}/${MAX_CONCURRENT_BATCHES} active batches)`);

  // Increment active batch count
  activeBatchCount++;

  // Send batch to background script for scanning
  chrome.runtime.sendMessage({
    action: "scanUrlsBatch",
    urlData: {
      pageUrl: window.location.href,
      urls: currentBatch
    }
  }, response => {
    // Decrement active batch count
    activeBatchCount--;

    // Handle response from the background script
    if (response && response.success) {
      debug(`Received results for ${response.results.length} URLs:`, response.results);
      // Reset retry counter on success
      retryAttempts = 0;

      // Process results
      if (Array.isArray(response.results)) {
        response.results.forEach(result => {
          // Make sure we have a valid result object
          if (result && typeof result.url === 'string') {
            // Get the normalized URL to use as the cache key
            const normalizedUrl = normalizeUrl(result.url);

            // Determine if the URL is malicious
            const isMalicious = !!result.isMalicious; // Convert to boolean

            debug(`URL ${result.url} is malicious: ${isMalicious}`);

            // Cache the result
            urlScanCache.set(normalizedUrl, isMalicious);

            // If we have the link element cached from visible queue, apply directly
            if (linkElements.has(normalizedUrl)) {
              applyResultToLink(linkElements.get(normalizedUrl), isMalicious);
            } else {
              // Otherwise find all matching links on the page
              const links = document.querySelectorAll(`a[href]`);
              links.forEach(link => {
                const normalizedLinkUrl = normalizeUrl(link.href);
                if (normalizedLinkUrl === normalizedUrl) {
                  applyResultToLink(link, isMalicious);
                }
              });
            }
          } else {
            console.warn("Invalid result in API response:", result);
          }
        });
      } else {
        console.error("Expected results array but got:", response.results);
      }
    } else {
      console.error("Error scanning URLs:", response?.error || "Unknown error", response);

      // Implement exponential backoff retry
      if (retryAttempts < MAX_RETRY_ATTEMPTS) {
        retryAttempts++;
        const delay = RETRY_DELAY_BASE * Math.pow(2, retryAttempts - 1);
        console.log(`Retry attempt ${retryAttempts}, waiting ${delay}ms before retrying...`);

        // Put URLs back in the queue
        currentBatch.forEach(url => queueUrlForScanning(url));

        // Set timer for retry with exponential backoff
        setTimeout(() => {
          processBatchScan();
        }, delay);
        return;
      } else {
        console.error("Maximum retry attempts reached, giving up on this batch");
        retryAttempts = 0;
      }
    }

    // If there are more URLs in either queue, process the next batch
    if (visibleUrlQueue.length > 0 || urlScanQueue.length > 0) {
      console.log(`${visibleUrlQueue.length} visible URLs and ${urlScanQueue.length} regular URLs still in queue`);

      // Process next batch with a small delay
      setTimeout(() => {
        processBatchScan();
      }, 50);
    }
  });
}

// Apply scan result to a link element
function applyResultToLink(linkElement, isMalicious) {
  // Force ignore specific domains regardless of scan result
  try {
    if (linkElement && linkElement.href) {
      const url = new URL(linkElement.href);
      if (url.host === '10.170.8.90:5000' || url.hostname === '10.170.8.90') {
        console.log("Forced ignoring malicious status for excluded IP:", url.host);
        return; // Skip applying any warnings
      }
    }
  } catch (e) {
    console.error("Error checking URL in applyResultToLink:", e);
  }

  debug(`Applying result to link: ${linkElement.href}, isMalicious: ${isMalicious}`);

  if (isMalicious) {
    // Mark as malicious
    linkElement.classList.add('secure-go-malicious-link');

    // Add warning icon before the link
    if (!linkElement.querySelector('.secure-go-warning-icon')) {
      const warningIcon = document.createElement('span');
      warningIcon.classList.add('secure-go-warning-icon');
      warningIcon.textContent = '⚠️ ';
      warningIcon.title = 'Potentially malicious URL';
      linkElement.insertBefore(warningIcon, linkElement.firstChild);
    }

    // Add warning tooltip
    linkElement.title = 'Warning: This link may be malicious';

    // Add click event to show warning
    if (!linkElement.dataset.secureGoWarningAdded) {
      linkElement.dataset.secureGoWarningAdded = 'true';
      linkElement.addEventListener('click', function (event) {
        // Skip warning for excluded domains
        try {
          const url = new URL(linkElement.href);
          if (url.host === '10.170.8.90:5000' || url.hostname === '10.170.8.90') {
            return; // Don't show warning
          }
        } catch (e) {}

        const confirmNavigation = confirm('Warning: This link may be malicious. Do you want to continue?');
        if (!confirmNavigation) {
          event.preventDefault();
        }
      });
    }
  }
}

// Check if a URL is valid for scanning (optimized)
function isValidUrl(url) {
  // Skip invalid URLs with fast checks first
  if (!url) return false;

  // Quick check for common non-http protocols
  const invalidProtocols = ['javascript:', 'mailto:', 'tel:', 'data:', 'blob:', '#'];
  for (const protocol of invalidProtocols) {
    if (url.startsWith(protocol)) return false;
  }

  try {
    // Try to parse the URL to make sure it's valid (last resort)
    new URL(url);
    return true;
  } catch (error) {
    return false;
  }
}

// Add CSS styles for malicious links
function addStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .secure-go-malicious-link {
      color: #dd2c00 !important;
      text-decoration: underline wavy #dd2c00 !important;
      background-color: rgba(255, 0, 0, 0.05) !important;
      padding: 2px 4px !important;
      border-radius: 2px !important;
    }
    
    .secure-go-warning-icon {
      margin-right: 4px;
    }
  `;
  document.head.appendChild(style);
}

// Check if current domain should be excluded from scanning
function shouldExcludeDomain() {
  try {
    const hostname = window.location.hostname;
    const hostWithPort = window.location.host; // includes port if present
    
    return EXCLUDED_DOMAINS.some(domain => 
      hostname === domain || 
      hostWithPort === domain || 
      hostname.endsWith('.' + domain)
    );
  } catch (e) {
    console.error("Error checking excluded domain:", e);
    return false;
  }
}

// Start the URL scanner only if enabled and not on excluded domains
if (URL_SCANNER_ENABLED && !shouldExcludeDomain()) {
  // Initialize styles
  addStyles();
  
  // Start the URL scanner
  initializeUrlScanner();
} else {
  console.log("URL scanner disabled by configuration or on excluded domain.");
} 