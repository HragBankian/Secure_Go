document.addEventListener('DOMContentLoaded', function() {
  // Get toggle elements
  const phishingToggle = document.getElementById('phishing-toggle');
  const urlScannerToggle = document.getElementById('url-scanner-toggle');
  
  // Load saved settings
  chrome.storage.sync.get(['phishingEnabled', 'urlScannerEnabled'], function(result) {
    // Set phishing toggle state
    if (result.phishingEnabled !== undefined) {
      phishingToggle.checked = result.phishingEnabled;
    } else {
      // Default to enabled
      phishingToggle.checked = true;
      chrome.storage.sync.set({ phishingEnabled: true });
    }
    
    // Set URL scanner toggle state
    if (result.urlScannerEnabled !== undefined) {
      urlScannerToggle.checked = result.urlScannerEnabled;
    } else {
      // Default to enabled
      urlScannerToggle.checked = true;
      chrome.storage.sync.set({ urlScannerEnabled: true });
    }
  });
  
  // Handle phishing toggle changes
  phishingToggle.addEventListener('change', function() {
    chrome.storage.sync.set({ phishingEnabled: this.checked });
    console.log('Phishing detection ' + (this.checked ? 'enabled' : 'disabled'));
  });
  
  // Handle URL scanner toggle changes
  urlScannerToggle.addEventListener('change', function() {
    chrome.storage.sync.set({ urlScannerEnabled: this.checked });
    console.log('URL scanner ' + (this.checked ? 'enabled' : 'disabled'));
  });
}); 