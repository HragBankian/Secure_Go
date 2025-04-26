// SecureGo - Popup script
// Handles the extension popup UI and interactions

document.addEventListener('DOMContentLoaded', function () {
  // Initialize UI elements
  const phishingToggle = document.getElementById('phishing-toggle');
  const refreshButton = document.getElementById('refresh-status');
  const openDashboardButton = document.getElementById('open-dashboard');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');

  // Statistics elements
  const emailsScanned = document.getElementById('emails-scanned');
  const phishingDetected = document.getElementById('phishing-detected');
  const detectionRate = document.getElementById('detection-rate');

  // API configuration
  const API_CONFIG = {
    baseUrl: "http://localhost:5000",
    endpoints: {
      health: "/health",
      dashboard: "/dashboard"
    }
  };

  // Load saved settings
  loadSettings();

  // Check API status on popup open
  checkApiStatus();

  // Load statistics
  loadStatistics();

  // Event listeners
  phishingToggle.addEventListener('change', function () {
    // Save setting to storage
    chrome.storage.sync.set({ 'phishingEnabled': this.checked }, function () {
      console.log('Phishing detection setting saved:', phishingToggle.checked);
    });
  });

  refreshButton.addEventListener('click', function () {
    checkApiStatus();
  });

  openDashboardButton.addEventListener('click', function () {
    // Open the dashboard in a new tab
    chrome.tabs.create({ url: `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.dashboard}` });
  });

  // Functions
  function loadSettings() {
    chrome.storage.sync.get(['phishingEnabled'], function (result) {
      // Default to enabled if not set
      phishingToggle.checked = result.phishingEnabled !== false;
    });
  }

  function checkApiStatus() {
    statusDot.className = 'status-dot warning';
    statusText.textContent = 'Checking...';

    fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.health}`)
      .then(response => response.json())
      .then(data => {
        if (data.status === 'ok') {
          if (data.model_loaded) {
            statusDot.className = 'status-dot online';
            statusText.textContent = 'API Online';
          } else {
            statusDot.className = 'status-dot warning';
            statusText.textContent = 'API Online (Model Not Loaded)';
          }
        } else {
          statusDot.className = 'status-dot offline';
          statusText.textContent = 'API Error';
        }
      })
      .catch(error => {
        console.error('API Health check failed:', error);
        statusDot.className = 'status-dot offline';
        statusText.textContent = 'API Offline';
      });
  }

  function loadStatistics() {
    // Try to load stats from storage
    chrome.storage.local.get(['emailStats'], function (result) {
      if (result.emailStats) {
        const stats = result.emailStats;
        emailsScanned.textContent = stats.totalScanned || 0;
        phishingDetected.textContent = stats.phishingDetected || 0;

        // Calculate detection rate
        if (stats.totalScanned > 0) {
          const rate = ((stats.phishingDetected / stats.totalScanned) * 100).toFixed(1);
          detectionRate.textContent = `${rate}%`;
        } else {
          detectionRate.textContent = '0%';
        }
      }
    });

    // Also try to fetch fresh stats from the API
    fetch(`${API_CONFIG.baseUrl}/stats`)
      .then(response => {
        if (response.ok) return response.json();
        throw new Error('Stats API not available');
      })
      .then(data => {
        // Update statistics with fresh data from API
        if (data && data.total_emails) {
          emailsScanned.textContent = data.total_emails;
          phishingDetected.textContent = data.phishing_emails;
          detectionRate.textContent = data.detection_rate;

          // Save to local storage for future reference
          chrome.storage.local.set({
            'emailStats': {
              totalScanned: data.total_emails,
              phishingDetected: data.phishing_emails,
              lastUpdated: Date.now()
            }
          });
        }
      })
      .catch(error => {
        console.log('Could not fetch fresh stats:', error);
      });
  }
}); 