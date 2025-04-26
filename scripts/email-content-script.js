// Content script for email phishing detection
// Extracts email content and sends it to the backend for analysis

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
  if (!emailData) return;

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

  // Show on-page overlay for debugging
  showScanOverlay(scanCounter, emailData.subject);

  // Check user setting and send for scanning
  chrome.storage.sync.get(['phishingEnabled'], function (result) {
    if (result.phishingEnabled !== false) {
      chrome.runtime.sendMessage({ action: 'scanEmail', emailData }, response => {
        console.log(`Scan #${emailData.scanId}: Received response:`, response);
        if (response && response.status === 'success') {
          displayEmailClassificationNotification(emailData, response.result);
        }
      });
    } else {
      console.log(`Scan #${scanCounter}: Scanning disabled by user`);
    }
  });
}

// Display a notification with the email classification (ham or spam)
function displayEmailClassificationNotification(emailData, scanResult) {
  const scanId = emailData.scanId;

  // Create notification element
  const notificationBanner = document.createElement('div');
  notificationBanner.className = 'securego-email-notification';

  // Show scan ID at the top
  if (scanId) {
    const scanIndicator = document.createElement('div');
    scanIndicator.style.cssText = 'font-size:12px; opacity:0.8; margin-bottom:4px;';
    scanIndicator.textContent = `Scan #${scanId}`;
    notificationBanner.appendChild(scanIndicator);
  }

  // Determine if it's phishing or legitimate
  const isPhishing = scanResult.result === "spam";

  notificationBanner.style.cssText += `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background-color: ${isPhishing ? '#f44336' : '#4CAF50'};
    color: white;
    padding: 10px 20px;
    font-family: Arial, sans-serif;
    font-size: 16px;
    text-align: center;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
  `;

  // Notification content with email subject
  const notificationContent = document.createElement('div');
  const subjectText = emailData.subject ? emailData.subject : '(no subject)';
  if (isPhishing) {
    notificationContent.innerHTML = `
      <strong>⚠️ Phishing Alert:</strong> Email <em>"${subjectText}"</em> appears suspicious and may be a phishing attempt.
    `;
  } else {
    notificationContent.innerHTML = `
      <strong>✅ Legitimate Email:</strong> Email <em>"${subjectText}"</em> has been analyzed and appears to be legitimate.
    `;
  }
  notificationBanner.appendChild(notificationContent);

  // Close button
  const closeButton = document.createElement('button');
  closeButton.textContent = '✕';
  closeButton.style.cssText = `
    align-self: flex-end;
    background: none;
    border: none;
    color: white;
    font-size: 20px;
    cursor: pointer;
    margin-top: 5px;
  `;
  closeButton.onclick = function () {
    document.body.removeChild(notificationBanner);
  };
  notificationBanner.appendChild(closeButton);

  // Auto-hide after 5 seconds for legitimate emails
  if (!isPhishing) {
    setTimeout(() => {
      if (document.body.contains(notificationBanner)) {
        document.body.removeChild(notificationBanner);
      }
    }, 5000);
  }

  // Remove existing notifications first
  const existing = document.querySelectorAll('.securego-email-notification');
  existing.forEach(n => document.body.removeChild(n));

  // Add the notification to the page
  document.body.appendChild(notificationBanner);
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
      // Updated Gmail extraction logic with multiple selector options
      const subjectSelectors = ['.ha h2', '.hP', '.g2', 'h2.hP'];
      const senderSelectors = ['.gD', '.go', '.cf.gK', '.cf.ix', 'span[email]'];
      const contentSelectors = ['.a3s', '.a3s.aiL', '.ii.gt', '.gs', '.ajV.ajx'];

      // Try each subject selector
      for (const selector of subjectSelectors) {
        const element = document.querySelector(selector);
        if (element) {
          subject = element.textContent.trim();
          console.log("Found Gmail subject with selector:", selector);
          break;
        }
      }

      // Try each sender selector
      for (const selector of senderSelectors) {
        const element = document.querySelector(selector);
        if (element) {
          // Try to get email attribute first, fallback to text content
          sender = element.getAttribute('email') || element.textContent.trim();
          console.log("Found Gmail sender with selector:", selector);
          break;
        }
      }

      // Try each content selector
      for (const selector of contentSelectors) {
        const element = document.querySelector(selector);
        if (element) {
          // Use the HTML of the email container instead of plain text
          content = element.innerHTML.trim();
          console.log("Found Gmail content with selector:", selector, "length:", content.length);
          break;
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

// Start the extension
initialize(); 