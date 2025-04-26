// URL Scanner Content Script
// Extracts URLs from webpages and sends them to backend for scanning

// Initialize URL scanning functionality
function initializeUrlScanner() {
  console.log("SecureGo URL Scanner initialized");
  
  // Check if URL scanning is enabled in settings
  chrome.storage.sync.get(['urlScannerEnabled'], function(result) {
    if (result.urlScannerEnabled !== false) { // Default to enabled if not set
      // Extract and scan URLs on the page
      extractAndScanUrls();
      
      // Set up observer to detect dynamically added links
      setupLinkObserver();
    } else {
      console.log("URL scanning is disabled in settings");
    }
  });
}

// Function to extract all URLs from the current page
function extractAndScanUrls() {
  // Get all anchor elements on the page
  const linkElements = document.querySelectorAll('a[href]');
  
  // Extract URLs from the links
  const urls = Array.from(linkElements).map(link => {
    return {
      url: link.href,
      element: link, // Store reference to the DOM element
      text: link.textContent.trim()
    };
  });
  
  // Filter out empty URLs, javascript: URLs, and mailto: URLs
  const validUrls = urls.filter(item => {
    const url = item.url.toLowerCase();
    return url && 
           !url.startsWith('javascript:') && 
           !url.startsWith('mailto:') &&
           !url.startsWith('tel:') &&
           !url.startsWith('sms:') &&
           !url.startsWith('data:') &&
           !url.startsWith('#');
  });
  
  // Only proceed if we have valid URLs
  if (validUrls.length > 0) {
    console.log(`Found ${validUrls.length} URLs to scan`);
    
    // Prepare URL data for scanning
    const urlData = {
      pageUrl: window.location.href,
      links: validUrls.map(item => ({
        url: item.url,
        text: item.text
      }))
    };
    
    // Send URLs to background script for scanning
    chrome.runtime.sendMessage({
      action: "scanUrls",
      urlData: urlData
    }, response => {
      if (response && response.success && response.results) {
        // Apply results to modify link appearance
        applyUrlScanResults(validUrls, response.results);
      } else {
        console.error("Error scanning URLs:", response?.error || "Unknown error");
      }
    });
  } else {
    console.log("No valid URLs found on the page");
  }
}

// Set up observer to detect new links added to the page
function setupLinkObserver() {
  // Create a mutation observer to detect when new links are added
  const observer = new MutationObserver(mutations => {
    let newLinksAdded = false;
    
    for (const mutation of mutations) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        // Check if any added nodes contain links
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            // Check if the node itself is a link or contains links
            if ((node.tagName === 'A' && node.hasAttribute('href')) || 
                node.querySelectorAll('a[href]').length > 0) {
              newLinksAdded = true;
              break;
            }
          }
        }
      }
      
      if (newLinksAdded) break;
    }
    
    // If new links were added, scan them
    if (newLinksAdded) {
      console.log("New links detected on the page, rescanning...");
      extractAndScanUrls();
    }
  });
  
  // Start observing the document body for added links
  observer.observe(document.body, { 
    childList: true, 
    subtree: true 
  });
  
  console.log("Link observer started");
}

// Apply scan results to the links on the page
function applyUrlScanResults(links, results) {
  if (!results || !Array.isArray(results)) {
    console.error("Invalid results format");
    return;
  }
  
  console.log("Applying URL scan results to links");
  
  // Create a map of URL to result for faster lookup
  const resultMap = {};
  results.forEach(result => {
    if (result.url) {
      resultMap[result.url] = result.isMalicious;
    }
  });
  
  // Apply styling to links based on scan results
  links.forEach(link => {
    const url = link.url;
    const element = link.element;
    
    if (url in resultMap) {
      const isMalicious = resultMap[url];
      
      if (isMalicious) {
        // Style for malicious URLs
        element.style.color = 'red';
        element.style.borderBottom = '1px dashed red';
        
        // Add warning tooltip and data attribute
        element.setAttribute('title', 'Warning: This link may be unsafe');
        element.setAttribute('data-securego-flagged', 'true');
        
        // Add a warning icon
        addWarningIcon(element);
      } else {
        // Style for safe URLs (optional)
        element.setAttribute('data-securego-checked', 'true');
      }
    }
  });
}

// Add a warning icon next to malicious links
function addWarningIcon(linkElement) {
  // Check if icon is already added
  if (linkElement.querySelector('.securego-warning-icon')) {
    return;
  }
  
  // Create warning icon
  const warningIcon = document.createElement('span');
  warningIcon.className = 'securego-warning-icon';
  warningIcon.textContent = ' ⚠️';
  warningIcon.style.color = 'red';
  warningIcon.style.fontSize = '0.8em';
  warningIcon.style.marginLeft = '3px';
  
  // Add the icon after the link text
  linkElement.appendChild(warningIcon);
}

// Start the URL scanner
initializeUrlScanner(); 