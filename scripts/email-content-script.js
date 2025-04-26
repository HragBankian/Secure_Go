// Content script for email phishing detection
// Extracts email content and sends it to the backend for analysis

// API endpoint configuration
const API_BASE_URL = "http://localhost:5000";

// Configuration for supported email domains
const SUPPORTED_EMAIL_DOMAINS = [
  "mail.google.com",   // Gmail
  "outlook.live.com",  // Outlook
  "outlook.office.com", // Outlook Office
  "outlook.office365.com", // Office 365
  "mail.yahoo.com"     // Yahoo Mail
];

// Add a global counter for scan invocations
let scanCounter = 0;
let lastSignature = null;

// Main initialization function
function initialize() {
  console.log("SecureGo email scanner initialized");

  // Check if current site is a supported email provider
  const currentDomain = window.location.hostname;
  if (!SUPPORTED_EMAIL_DOMAINS.includes(currentDomain)) {
    console.log("Not a supported email provider. Email scanning disabled.");
    return;
  }

  // One-time scan on initial page load
  extractAndSendEmailContent();

  // Set up mutation observer to detect when emails are opened
  setupEmailObserver();
}

// Set up observer to monitor DOM changes for email view
function setupEmailObserver() {
  // Select the appropriate container based on the email provider
  const emailContainer = getEmailContainerSelector();
  if (!emailContainer) {
    console.log("Could not identify email container for this provider");
    return;
  }

  console.log("Setting up observer for email container:", emailContainer);

  // Create a mutation observer to detect when emails are opened/loaded
  const observer = new MutationObserver((mutations) => {
    console.log("DOM mutation detected:", mutations.length, "changes");
    // When DOM changes, check if email content has changed
    extractAndSendEmailContent();
  });

  // Wait for the email container to be available in the DOM
  waitForElement(emailContainer).then(container => {
    // Start observing the container for changes
    observer.observe(container, { childList: true, subtree: true });
    console.log("Email observer started on container:", container);

    // Also immediately try to extract email content
    extractAndSendEmailContent();
  }).catch(error => {
    console.error("Error setting up email observer:", error);
  });

  // Also scan emails when the page loads (for already opened emails)
  setTimeout(extractAndSendEmailContent, 2000);

  // For Gmail specifically, add click handler for email items
  if (window.location.hostname === "mail.google.com") {
    document.addEventListener('click', function (e) {
      // Add a small delay to allow the email to load
      setTimeout(extractAndSendEmailContent, 1000);
    });
  }
}

// Get the appropriate container selector based on the email provider
function getEmailContainerSelector() {
  const domain = window.location.hostname;

  if (domain === "mail.google.com") {
    // Gmail message body container
    // Try multiple selectors for Gmail to be more robust
    return ".aDP, .aeF, .nH";  // Multiple possible Gmail container selectors
  } else if (domain.includes("outlook")) {
    return ".ReadingPaneContent";  // Outlook email reading pane
  } else if (domain === "mail.yahoo.com") {
    return ".message-view";  // Yahoo Mail message view
  }

  return null;
}

// Helper function to wait for an element to be available in the DOM
function waitForElement(selector, timeout = 10000) {
  return new Promise((resolve, reject) => {
    const element = document.querySelector(selector);
    if (element) {
      return resolve(element);
    }

    const startTime = Date.now();
    const observer = new MutationObserver(() => {
      const element = document.querySelector(selector);
      if (element) {
        observer.disconnect();
        resolve(element);
      } else if (Date.now() - startTime > timeout) {
        observer.disconnect();
        reject(new Error(`Timeout waiting for element: ${selector}`));
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
  });
}

// Small overlay to show scan status
function showScanOverlay(scanId, subject) {
  let overlay = document.getElementById('securego-scan-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'securego-scan-overlay';
    overlay.style.cssText = 'position:fixed;bottom:10px;left:10px;background:rgba(0,0,0,0.7);color:#fff;padding:5px 10px;border-radius:4px;font-size:12px;z-index:10000;';
    document.body.appendChild(overlay);
  }
  overlay.textContent = `Scan #${scanId}: ${subject || '(no subject)'}`;
  // Auto-hide after a few seconds
  clearTimeout(overlay._timeout);
  overlay._timeout = setTimeout(() => {
    if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
  }, 4000);
}

// Extract and send email content to the backend
function extractAndSendEmailContent() {
  const emailData = extractEmailData();
  if (!emailData) {
    console.error("Failed to extract email data");
    return;
  }

  // Create a unique signature to detect if this email has changed
  const signature = `${emailData.sender}|${emailData.subject}`;
  if (signature === lastSignature) {
    console.log("Skipping duplicate email scan (signature unchanged)");
    return;
  }
  lastSignature = signature;

  // Increment and attach a scan ID
  scanCounter++;
  emailData.scanId = scanCounter;
  console.log(`Scan #${scanCounter}: Extracted email data for subject: "${emailData.subject || '(no subject)'}"`);
  
  // Extra debug info for content length
  console.log(`Scan #${scanCounter}: Content length: ${emailData.content ? emailData.content.length : 0} bytes`);

  // Show on-page overlay for debugging
  showScanOverlay(scanCounter, emailData.subject);

  // Check user setting and send for scanning
  chrome.storage.sync.get(['phishingEnabled'], function (result) {
    if (result.phishingEnabled !== false) {
      console.log(`Scan #${scanCounter}: Sending to background script for API scan...`);
      
      handleEmailAnalysis(emailData);
    } else {
      console.log(`Scan #${scanCounter}: Scanning disabled by user`);
    }
  });
}

// Display notifications
function createNotificationContainer() {
  // Remove any existing notification containers
  const existingContainer = document.getElementById('securemail-notification-container');
  if (existingContainer) {
    existingContainer.remove();
  }

  // Create a new container
  const container = document.createElement('div');
  container.id = 'securemail-notification-container';
  container.style.position = 'fixed';
  container.style.top = '20px';
  container.style.right = '20px';
  container.style.zIndex = '9999';
  container.style.fontFamily = 'Arial, sans-serif';
  document.body.appendChild(container);
  return container;
}

function displayLoadingNotification() {
  const container = createNotificationContainer();
  
  const notification = document.createElement('div');
  notification.id = 'securemail-loading-notification';
  notification.style.backgroundColor = '#ffffff';
  notification.style.color = '#333333';
  notification.style.padding = '15px 20px';
  notification.style.borderRadius = '8px';
  notification.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
  notification.style.marginBottom = '10px';
  notification.style.display = 'flex';
  notification.style.alignItems = 'center';
  notification.style.minWidth = '300px';
  notification.style.transition = 'all 0.3s ease';
  notification.style.border = '1px solid #e0e0e0';
  
  // Create spinner
  const spinner = document.createElement('div');
  spinner.style.width = '20px';
  spinner.style.height = '20px';
  spinner.style.borderRadius = '50%';
  spinner.style.border = '3px solid #f3f3f3';
  spinner.style.borderTop = '3px solid #3498db';
  spinner.style.animation = 'spin 1s linear infinite';
  spinner.style.marginRight = '10px';
  
  // Add keyframe animation for spinner
  const style = document.createElement('style');
  style.innerHTML = `
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
  
  const text = document.createElement('div');
  text.textContent = 'Analyzing email for security threats...';
  text.style.flex = '1';
  
  notification.appendChild(spinner);
  notification.appendChild(text);
  container.appendChild(notification);
}

function hideLoadingNotification() {
  const loadingNotification = document.getElementById('securemail-loading-notification');
  if (loadingNotification) {
    loadingNotification.remove();
  }
}

function displayResultNotification(message, isWarning = false) {
  const container = createNotificationContainer();
  
  const notification = document.createElement('div');
  notification.id = 'securemail-result-notification';
  notification.style.backgroundColor = isWarning ? '#fee2e2' : '#ecfdf5';
  notification.style.color = isWarning ? '#991b1b' : '#065f46';
  notification.style.padding = '15px 20px';
  notification.style.borderRadius = '8px';
  notification.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
  notification.style.marginBottom = '10px';
  notification.style.display = 'flex';
  notification.style.alignItems = 'center';
  notification.style.minWidth = '300px';
  notification.style.maxWidth = '400px';
  notification.style.transition = 'all 0.3s ease';
  notification.style.border = `1px solid ${isWarning ? '#fca5a5' : '#a7f3d0'}`;
  
  const icon = document.createElement('div');
  icon.innerHTML = isWarning 
    ? '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>' 
    : '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>';
  icon.style.marginRight = '10px';
  icon.style.flexShrink = '0';
  
  const textContainer = document.createElement('div');
  textContainer.style.flex = '1';
  
  const text = document.createElement('div');
  text.innerHTML = message;
  text.style.marginBottom = '8px';
  
  const actions = document.createElement('div');
  actions.style.display = 'flex';
  actions.style.justifyContent = 'space-between';
  actions.style.marginTop = '10px';
  
  // Add buttons for actions
  if (isWarning) {
    const reportBtn = document.createElement('button');
    reportBtn.textContent = 'Report Phishing';
    reportBtn.style.backgroundColor = '#dc2626';
    reportBtn.style.color = 'white';
    reportBtn.style.border = 'none';
    reportBtn.style.padding = '5px 10px';
    reportBtn.style.borderRadius = '4px';
    reportBtn.style.cursor = 'pointer';
    reportBtn.style.marginRight = '8px';
    reportBtn.onclick = () => {
      // Functionality to report phishing
      notification.innerHTML = '<div style="text-align: center;">Thank you for reporting this email.</div>';
      setTimeout(() => {
        notification.remove();
      }, 2000);
    };
    actions.appendChild(reportBtn);
  }
  
  const dismissBtn = document.createElement('button');
  dismissBtn.textContent = 'Dismiss';
  dismissBtn.style.backgroundColor = '#e5e7eb';
  dismissBtn.style.color = '#374151';
  dismissBtn.style.border = 'none';
  dismissBtn.style.padding = '5px 10px';
  dismissBtn.style.borderRadius = '4px';
  dismissBtn.style.cursor = 'pointer';
  dismissBtn.onclick = () => notification.remove();
  actions.appendChild(dismissBtn);
  
  textContainer.appendChild(text);
  textContainer.appendChild(actions);
  
  notification.appendChild(icon);
  notification.appendChild(textContainer);
  container.appendChild(notification);
  
  // Auto-dismiss after 10 seconds if not a warning
  if (!isWarning) {
    setTimeout(() => {
      if (notification && notification.parentNode) {
        notification.remove();
      }
    }, 10000);
  }
}

function displayErrorNotification(message) {
  const container = createNotificationContainer();
  
  const notification = document.createElement('div');
  notification.id = 'securemail-error-notification';
  notification.style.backgroundColor = '#fff1f2';
  notification.style.color = '#881337';
  notification.style.padding = '15px 20px';
  notification.style.borderRadius = '8px';
  notification.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
  notification.style.marginBottom = '10px';
  notification.style.display = 'flex';
  notification.style.alignItems = 'center';
  notification.style.minWidth = '300px';
  notification.style.transition = 'all 0.3s ease';
  notification.style.border = '1px solid #fecdd3';
  
  const icon = document.createElement('div');
  icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>';
  icon.style.marginRight = '10px';
  
  const text = document.createElement('div');
  text.textContent = message;
  text.style.flex = '1';
  
  const closeBtn = document.createElement('div');
  closeBtn.innerHTML = '&times;';
  closeBtn.style.cursor = 'pointer';
  closeBtn.style.marginLeft = '10px';
  closeBtn.style.fontSize = '20px';
  closeBtn.style.color = '#881337';
  closeBtn.onclick = () => notification.remove();
  
  notification.appendChild(icon);
  notification.appendChild(text);
  notification.appendChild(closeBtn);
  container.appendChild(notification);
  
  // Auto-dismiss after 5 seconds
  setTimeout(() => {
    if (notification && notification.parentNode) {
      notification.remove();
    }
  }, 5000);
}

// Extract email data based on the email provider
function extractEmailData() {
  const domain = window.location.hostname;
  console.log("Attempting to extract email data from:", domain);

  try {
    let sender = "";
    let subject = "";
    let content = "";

    // Extract email data based on the provider
    if (domain === "mail.google.com") {
      // Updated Gmail extraction logic with more modern selectors
      const subjectSelectors = [
        '.ha h2', 
        '.hP', 
        '.g2', 
        'h2.hP', 
        '.nH .hP',
        'h2.h4',    // Modern Gmail
        'div[role="heading"]', // Role-based selector
        '.UI h2',    // Some Gmail views
        'div.xY h2'  // Enhanced Gmail view
      ];
      
      const senderSelectors = [
        '.gD', 
        '.go', 
        '.cf.gK', 
        '.cf.ix', 
        'span[email]',
        '.gI span',
        '.gE div', // Modern sender line 
        '.gF div', // Alternative modern sender
        'div[email]', // Email attribute
        '.gb_pd' // New Gmail UI
      ];
      
      const contentSelectors = [
        '.a3s', 
        '.a3s.aiL', 
        '.ii.gt', 
        '.gs', 
        '.ajV.ajx',
        'div[role="region"]', // Role-based content area
        '.aQH',   // Modern Gmail content
        '.adn',   // Common content container
        '.adP',   // Extra container
        '.adO'    // Alternative content area
      ];

      // Try each subject selector
      for (const selector of subjectSelectors) {
        const elements = document.querySelectorAll(selector);
        for (const element of elements) {
          if (element && element.textContent.trim()) {
            subject = element.textContent.trim();
            console.log("Found Gmail subject with selector:", selector);
            break;
          }
        }
        if (subject) break;
      }

      // Try each sender selector
      for (const selector of senderSelectors) {
        const elements = document.querySelectorAll(selector);
        for (const element of elements) {
          // Try to get email attribute first, fallback to text content
          const potentialSender = element.getAttribute('email') || element.textContent.trim();
          if (potentialSender) {
            sender = potentialSender;
            console.log("Found Gmail sender with selector:", selector);
            break;
          }
        }
        if (sender) break;
      }

      // Try each content selector
      for (const selector of contentSelectors) {
        const elements = document.querySelectorAll(selector);
        for (const element of elements) {
          if (element) {
            // Use the HTML of the email container instead of plain text
            const potentialContent = element.innerHTML.trim();
            if (potentialContent && potentialContent.length > 100) { // Ensure it has substantial content
              content = potentialContent;
              console.log("Found Gmail content with selector:", selector, "length:", content.length);
              break;
            }
          }
        }
        if (content) break;
      }

      // If still no content, try a more aggressive approach
      if (!content) {
        console.log("Attempting more aggressive content extraction for Gmail");
        
        // Look for any large content blocks
        const allDivs = document.querySelectorAll('div');
        let largestContentDiv = null;
        let largestLength = 0;
        
        for (const div of allDivs) {
          const html = div.innerHTML;
          if (html && html.length > largestLength && html.length > 500) {
            largestLength = html.length;
            largestContentDiv = div;
          }
        }
        
        if (largestContentDiv) {
          content = largestContentDiv.innerHTML;
          console.log("Found large content block with length:", content.length);
        }
      }

    } else if (domain.includes("outlook")) {
      // Outlook extraction logic
      const subjectElement = document.querySelector('.SubjectLine');
      subject = subjectElement ? subjectElement.textContent.trim() : "";

      const senderElement = document.querySelector('.conductorContent .FsLinkButton');
      sender = senderElement ? senderElement.textContent.trim() : "";

      const contentElement = document.querySelector('.ReadingPaneContent');
      // Pass the container's HTML for analysis
      content = contentElement ? contentElement.innerHTML.trim() : "";

    } else if (domain === "mail.yahoo.com") {
      // Yahoo Mail extraction logic
      const subjectElement = document.querySelector('.subject-container');
      subject = subjectElement ? subjectElement.textContent.trim() : "";

      const senderElement = document.querySelector('.from span');
      sender = senderElement ? senderElement.textContent.trim() : "";

      const contentElement = document.querySelector('.message-body');
      // Send HTML content of the message body
      content = contentElement ? contentElement.innerHTML.trim() : "";
    }

    // Debug the extracted information
    console.log("Extracted email data:", {
      subject: subject ? `${subject.substring(0, 20)}...` : "(none)",
      sender: sender || "(none)",
      contentLength: content.length
    });

    // If we didn't find any content with selectors, fallback to entire page HTML
    if (!content) {
      console.warn("No email-specific content found; falling back to full page HTML.");
      content = document.documentElement.outerHTML;
    }

    // Debug the final content length
    console.log("Final content length passed to API:", content.length);

    return { sender, subject, content };
  } catch (error) {
    console.error("Error extracting email data:", error);
  }

  return null;
}

// Send the email data to the API for analysis
async function handleEmailAnalysis(emailData) {
  console.log("Starting email analysis...");

  try {
    // Check if we have substantive content
    if (!emailData.content || emailData.content.length < 200) {
      console.warn("Email content is too short for reliable analysis:", emailData.content?.length || 0, "chars");
      displayErrorNotification("Email content is too short for reliable analysis.");
      return;
    }

    console.log("Preparing API request with extracted email data");
    // Show loading notification
    displayLoadingNotification();

    // Prepare the request
    const requestBody = {
      sender: emailData.sender,
      subject: emailData.subject,
      content: emailData.content
    };

    // Set API endpoint and request options
    const apiUrl = API_BASE_URL + "/AI";
    
    const requestOptions = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(requestBody)
    };

    // Make the API request with retries
    let response = null;
    let retries = 0;
    const maxRetries = 2;
    
    while (retries <= maxRetries) {
      try {
        console.log(`API request attempt ${retries + 1}/${maxRetries + 1}`);
        response = await fetch(apiUrl, requestOptions);
        
        if (response.ok) {
          break; // Success, exit retry loop
        } else {
          console.warn(`API responded with status ${response.status}`);
          // If server error, retry; if client error, don't
          if (response.status >= 500) {
            retries++;
            if (retries <= maxRetries) {
              const backoffTime = Math.pow(2, retries) * 1000; // Exponential backoff
              console.log(`Retrying in ${backoffTime}ms...`);
              await new Promise(resolve => setTimeout(resolve, backoffTime));
            }
          } else {
            // Client error, don't retry
            break;
          }
        }
      } catch (error) {
        console.error("Network error during API request:", error);
        retries++;
        if (retries <= maxRetries) {
          const backoffTime = Math.pow(2, retries) * 1000;
          console.log(`Network error, retrying in ${backoffTime}ms...`);
          await new Promise(resolve => setTimeout(resolve, backoffTime));
        } else {
          break;
        }
      }
    }

    // Handle the API response
    if (response && response.ok) {
      try {
        const data = await response.json();
        console.log("Email analysis response received:", data);
        
        // Process the results and display notification
        if (data && data.result) {
          // Parse confidence value to ensure it's a proper percentage
          const confidence = parseFloat(data.confidence);
          const formattedConfidence = isNaN(confidence) ? "unknown" : `${(confidence * 100).toFixed(1)}%`;
          
          // Use the new notification system
          if (data.result === "spam") {
            displayResultNotification(
              `🚨 This email is likely phishing (confidence: ${formattedConfidence})`,
              true
            );
          } else {
            displayResultNotification(
              `✅ This email appears to be legitimate (confidence: ${formattedConfidence})`,
              false
            );
          }
          
          // Log analysis details for debugging
          console.log(`Classification: ${data.result}, Confidence: ${formattedConfidence}`);
        } else {
          throw new Error("Invalid response format");
        }
      } catch (parseError) {
        console.error("Error parsing API response:", parseError);
        displayErrorNotification("Error processing analysis results.");
      }
    } else {
      // Handle non-OK response
      let errorMsg = "Failed to analyze email.";
      
      try {
        // Try to get more detailed error message from response
        if (response) {
          const errorData = await response.text();
          console.error("API error response:", errorData);
          try {
            const parsedError = JSON.parse(errorData);
            if (parsedError && parsedError.error) {
              errorMsg = parsedError.error;
            }
          } catch (e) {
            // If parsing fails, use status text
            errorMsg = response.statusText || errorMsg;
          }
        }
      } catch (e) {
        console.error("Error extracting error details:", e);
      }
      
      displayErrorNotification(`${errorMsg} Please try again.`);
    }
  } catch (error) {
    console.error("Error in handleEmailAnalysis:", error);
    displayErrorNotification("An unexpected error occurred while analyzing the email.");
  } finally {
    hideLoadingNotification();
  }
}

// Start the extension
initialize(); 