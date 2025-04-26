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

// Main initialization function
function initialize() {
  console.log("SecureGo email scanner initialized");
  
  // Check if current site is a supported email provider
  const currentDomain = window.location.hostname;
  if (!SUPPORTED_EMAIL_DOMAINS.includes(currentDomain)) {
    console.log("Not a supported email provider. Email scanning disabled.");
    return;
  }
  
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
  
  // Create a mutation observer to detect when emails are opened/loaded
  const observer = new MutationObserver(() => {
    // When DOM changes, check if email content has changed
    extractAndSendEmailContent();
  });
  
  // Wait for the email container to be available in the DOM
  waitForElement(emailContainer).then(container => {
    // Start observing the container for changes
    observer.observe(container, { childList: true, subtree: true });
    console.log("Email observer started");
  }).catch(error => {
    console.error("Error setting up email observer:", error);
  });
  
  // Also scan emails when the page loads (for already opened emails)
  setTimeout(extractAndSendEmailContent, 2000);
}

// Get the appropriate container selector based on the email provider
function getEmailContainerSelector() {
  const domain = window.location.hostname;
  
  if (domain === "mail.google.com") {
    return ".AO";  // Gmail main content container
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

// Extract and send email content to the backend
function extractAndSendEmailContent() {
  const emailData = extractEmailData();
  if (emailData) {
    console.log("Extracted email data:", emailData);
    
    // Check if phishing detection is enabled in extension settings
    chrome.storage.sync.get(['phishingEnabled'], function(result) {
      if (result.phishingEnabled !== false) {  // Default to enabled if not set
        // Send data to background script to forward to the backend
        chrome.runtime.sendMessage({
          action: "scanEmail",
          emailData: emailData
        }, response => {
          console.log("Email data sent to background script", response);
        });
      } else {
        console.log("Email scanning is disabled in settings");
      }
    });
  }
}

// Extract email data based on the email provider
function extractEmailData() {
  const domain = window.location.hostname;
  
  try {
    let sender = "";
    let subject = "";
    let content = "";
    
    // Extract email data based on the provider
    if (domain === "mail.google.com") {
      // Gmail extraction logic
      const headerElement = document.querySelector('.ha h2');
      subject = headerElement ? headerElement.textContent.trim() : "";
      
      const senderElement = document.querySelector('.gD');
      sender = senderElement ? senderElement.getAttribute('email') : "";
      
      const contentElement = document.querySelector('.a3s');
      content = contentElement ? contentElement.innerText.trim() : "";
      
    } else if (domain.includes("outlook")) {
      // Outlook extraction logic
      const subjectElement = document.querySelector('.SubjectLine');
      subject = subjectElement ? subjectElement.textContent.trim() : "";
      
      const senderElement = document.querySelector('.conductorContent .FsLinkButton');
      sender = senderElement ? senderElement.textContent.trim() : "";
      
      const contentElement = document.querySelector('.ReadingPaneContent');
      content = contentElement ? contentElement.innerText.trim() : "";
      
    } else if (domain === "mail.yahoo.com") {
      // Yahoo Mail extraction logic
      const subjectElement = document.querySelector('.subject-container');
      subject = subjectElement ? subjectElement.textContent.trim() : "";
      
      const senderElement = document.querySelector('.from span');
      sender = senderElement ? senderElement.textContent.trim() : "";
      
      const contentElement = document.querySelector('.message-body');
      content = contentElement ? contentElement.innerText.trim() : "";
    }
    
    // Only proceed if we have at least some content
    if (sender || content) {
      return { sender, subject, content };
    }
  } catch (error) {
    console.error("Error extracting email data:", error);
  }
  
  return null;
}

// Start the extension
initialize(); 