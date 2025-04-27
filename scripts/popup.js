// SecureGo - Popup script
// Handles the extension popup UI and interactions

document.addEventListener('DOMContentLoaded', function () {
  // Initialize UI elements
  const phishingToggle = document.getElementById('phishing-toggle');
  const nsfwToggle = document.getElementById('nsfw-toggle');
  const refreshButton = document.getElementById('refresh-status');
  const openDashboardButton = document.getElementById('open-dashboard');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');

  // Statistics elements
  const emailsScanned = document.getElementById('emails-scanned');
  const phishingDetected = document.getElementById('phishing-detected');
  const detectionRate = document.getElementById('detection-rate');
  
  // NSFW statistics elements
  const sitesAnalyzed = document.getElementById('sites-analyzed');
  const nsfwDetected = document.getElementById('nsfw-detected');
  const nsfwRecentList = document.getElementById('nsfw-recent-list');
  const nsfwDetectionRateElem = document.getElementById('nsfw-detection-rate');
  const nsfwBlockedCount = document.getElementById('nsfw-blocked-count');
  const nsfwDetailsLink = document.getElementById('nsfw-details-link');

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
  
  // Load NSFW statistics
  loadNsfwStatistics();

  // Event listeners
  phishingToggle.addEventListener('change', function () {
    // Save setting to storage
    chrome.storage.sync.set({ 'phishingEnabled': this.checked }, function () {
      console.log('Phishing detection setting saved:', phishingToggle.checked);
    });
  });
  
  nsfwToggle.addEventListener('change', function () {
    // Save setting to storage
    chrome.storage.sync.set({ 'nsfwCheckEnabled': this.checked }, function () {
      console.log('NSFW detection setting saved:', nsfwToggle.checked);
    });
  });

  refreshButton.addEventListener('click', function () {
    checkApiStatus();
    loadNsfwStatistics(); // Refresh NSFW stats too
  });

  openDashboardButton.addEventListener('click', function () {
    // Open the dashboard in a new tab
    chrome.tabs.create({ url: `${API_CONFIG.baseUrl}${API_CONFIG.endpoints.dashboard}` });
  });
  
  // Add dashboard link for NSFW details if available
  if (nsfwDetailsLink) {
    nsfwDetailsLink.addEventListener('click', function(e) {
      e.preventDefault();
      chrome.tabs.create({ url: chrome.runtime.getURL("html/nsfw-dashboard.html") });
    });
  }

  // Functions
  function loadSettings() {
    chrome.storage.sync.get(['phishingEnabled', 'nsfwCheckEnabled'], function (result) {
      // Default to enabled if not set
      phishingToggle.checked = result.phishingEnabled !== false;
      nsfwToggle.checked = result.nsfwCheckEnabled !== false;
    });
  }

  function checkApiStatus() {
    statusDot.className = 'status-dot warning';
    statusText.textContent = 'Checking...';

    fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.health}`)
      .then(response => response.json())
      .then(data => {
        if (data.status === 'ok' || data.status === 'healthy' || data.status === 'degraded') {
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
  
  function loadNsfwStatistics() {
    // Load NSFW stats from storage
    chrome.storage.local.get(['nsfwStats', 'nsfwUrlCache', 'nsfwSiteList', 'nsfwUserActions'], function (result) {
      let totalAnalyzed = 0;
      let totalNsfw = 0;
      let blockedCount = 0;
      
      // If we have dedicated stats
      if (result.nsfwStats) {
        totalAnalyzed = result.nsfwStats.totalAnalyzed || 0;
        totalNsfw = result.nsfwStats.totalNsfw || 0;
      } 
      // Otherwise compute from cache
      else if (result.nsfwUrlCache) {
        const cache = result.nsfwUrlCache;
        totalAnalyzed = Object.keys(cache).length;
        
        // Count NSFW entries
        for (const url in cache) {
          if (cache[url].isNsfw) {
            totalNsfw++;
          }
        }
      }
      
      // Update basic stats UI
      sitesAnalyzed.textContent = totalAnalyzed;
      nsfwDetected.textContent = totalNsfw;
      
      // Calculate and update detection rate if element exists
      if (nsfwDetectionRateElem && totalAnalyzed > 0) {
        const rate = ((totalNsfw / totalAnalyzed) * 100).toFixed(1);
        nsfwDetectionRateElem.textContent = `${rate}%`;
      }
      
      // Count warnings shown vs proceeded actions
      if (result.nsfwUserActions) {
        const warningShownCount = result.nsfwUserActions.filter(action => action.action === 'warning_shown').length;
        const proceedCount = result.nsfwUserActions.filter(action => action.action === 'proceed').length;
        
        // Calculate how many warnings were effective (warning shown but didn't proceed)
        blockedCount = warningShownCount - proceedCount;
        
        // Update UI if element exists
        if (nsfwBlockedCount) {
          nsfwBlockedCount.textContent = blockedCount;
        }
      }
      
      // Update recent NSFW sites list if element exists
      if (nsfwRecentList && result.nsfwSiteList) {
        updateRecentNsfwSites(result.nsfwSiteList);
      }
      
      // Save computed stats for consistency
      chrome.storage.local.set({
        'nsfwStats': {
          totalAnalyzed: totalAnalyzed,
          totalNsfw: totalNsfw,
          blockedCount: blockedCount,
          detectionRate: totalAnalyzed > 0 ? (totalNsfw / totalAnalyzed) : 0,
          lastUpdated: Date.now()
        }
      });
    });
  }
  
  // Update recent NSFW sites list
  function updateRecentNsfwSites(siteList) {
    // Get only NSFW sites and limit to 5 most recent
    const recentNsfwSites = siteList
      .filter(site => site.isNsfw)
      .slice(0, 5);
      
    // Clear current list
    nsfwRecentList.innerHTML = '';
    
    if (recentNsfwSites.length === 0) {
      // Show message if no NSFW sites found
      const emptyItem = document.createElement('li');
      emptyItem.textContent = 'No NSFW sites detected yet';
      emptyItem.className = 'nsfw-empty-list';
      nsfwRecentList.appendChild(emptyItem);
      return;
    }
    
    // Add each site to the list
    recentNsfwSites.forEach(site => {
      const listItem = document.createElement('li');
      listItem.className = 'nsfw-list-item';
      
      // Create domain text
      const domainSpan = document.createElement('span');
      domainSpan.className = 'nsfw-domain';
      domainSpan.textContent = site.domain;
      
      // Create date text
      const dateSpan = document.createElement('span');
      dateSpan.className = 'nsfw-date';
      dateSpan.textContent = site.nsfwDetectedOn || site.lastVisit;
      
      // Add to list item
      listItem.appendChild(domainSpan);
      listItem.appendChild(dateSpan);
      
      // Add tooltip
      listItem.title = `Detected: ${site.nsfwDetectedOn || 'Unknown'}\nVisits: ${site.visits}`;
      
      // Add to list
      nsfwRecentList.appendChild(listItem);
    });
  }
}); 