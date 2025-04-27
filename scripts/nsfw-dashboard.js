// SecureGo - NSFW Dashboard Script
// Handles displaying NSFW statistics and site information

document.addEventListener('DOMContentLoaded', function() {
  // Initialize UI elements
  const totalSitesElem = document.getElementById('total-sites');
  const nsfwSitesElem = document.getElementById('nsfw-sites');
  const detectionRateElem = document.getElementById('detection-rate');
  const blockedCountElem = document.getElementById('blocked-count');
  
  // Tables
  const activityTable = document.getElementById('activity-tbody');
  const activityEmptyState = document.getElementById('activity-empty');
  const nsfwSitesTable = document.getElementById('nsfw-sites-tbody');
  const nsfwSitesEmptyState = document.getElementById('nsfw-sites-empty');
  
  // Pagination
  const activityPagination = document.getElementById('activity-pagination');
  const nsfwPagination = document.getElementById('nsfw-pagination');
  
  // Buttons
  const refreshBtn = document.getElementById('refresh-btn');
  const clearDataBtn = document.getElementById('clear-data-btn');
  const exportBtn = document.getElementById('export-btn');
  
  // Pagination state
  const paginationState = {
    activity: {
      currentPage: 1,
      itemsPerPage: 10,
      totalPages: 1
    },
    nsfwSites: {
      currentPage: 1,
      itemsPerPage: 10,
      totalPages: 1
    }
  };
  
  // Load data when page loads
  loadDashboardData();
  
  // Button event listeners
  refreshBtn.addEventListener('click', loadDashboardData);
  
  clearDataBtn.addEventListener('click', function() {
    if (confirm('Are you sure you want to clear all NSFW detection data? This cannot be undone.')) {
      clearAllData();
    }
  });
  
  exportBtn.addEventListener('click', exportData);
  
  // Main function to load all dashboard data
  function loadDashboardData() {
    chrome.storage.local.get(['nsfwStats', 'nsfwSiteList', 'nsfwUserActions'], function(result) {
      // Update summary statistics
      updateSummaryStats(result);
      
      // Update activity table
      updateActivityTable(result.nsfwUserActions || []);
      
      // Update NSFW sites table
      updateNsfwSitesTable(result.nsfwSiteList || []);
    });
  }
  
  // Update summary statistics
  function updateSummaryStats(data) {
    const stats = data.nsfwStats || {
      totalAnalyzed: 0,
      totalNsfw: 0,
      blockedCount: 0,
      detectionRate: 0
    };
    
    // Set values
    totalSitesElem.textContent = stats.totalAnalyzed || 0;
    nsfwSitesElem.textContent = stats.totalNsfw || 0;
    
    // Calculate detection rate
    const detectionRate = stats.totalAnalyzed > 0 
      ? ((stats.totalNsfw / stats.totalAnalyzed) * 100).toFixed(1) 
      : '0.0';
    detectionRateElem.textContent = `${detectionRate}%`;
    
    // Set blocked count
    blockedCountElem.textContent = stats.blockedCount || 0;
  }
  
  // Update activity table
  function updateActivityTable(actions) {
    // Handle empty state
    if (!actions || actions.length === 0) {
      activityTable.innerHTML = '';
      activityEmptyState.style.display = 'block';
      activityPagination.innerHTML = '';
      return;
    }
    
    activityEmptyState.style.display = 'none';
    
    // Set up pagination
    const totalItems = actions.length;
    paginationState.activity.totalPages = Math.ceil(totalItems / paginationState.activity.itemsPerPage);
    
    // Validate current page
    if (paginationState.activity.currentPage > paginationState.activity.totalPages) {
      paginationState.activity.currentPage = 1;
    }
    
    // Calculate slice indices
    const startIndex = (paginationState.activity.currentPage - 1) * paginationState.activity.itemsPerPage;
    const endIndex = Math.min(startIndex + paginationState.activity.itemsPerPage, totalItems);
    
    // Get page data
    const pageData = actions.slice(startIndex, endIndex);
    
    // Clear table
    activityTable.innerHTML = '';
    
    // Populate table
    pageData.forEach(action => {
      const row = document.createElement('tr');
      
      // Format timestamp
      const date = new Date(action.timestamp);
      const formattedDate = `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
      
      // Create cells
      const timestampCell = document.createElement('td');
      timestampCell.textContent = formattedDate;
      
      const domainCell = document.createElement('td');
      domainCell.textContent = action.domain;
      
      const actionCell = document.createElement('td');
      actionCell.textContent = formatActionType(action.action);
      
      const statusCell = document.createElement('td');
      const statusTag = document.createElement('span');
      statusTag.className = 'tag ' + getActionStatusClass(action.action);
      statusTag.textContent = getActionStatusText(action.action);
      statusCell.appendChild(statusTag);
      
      // Add cells to row
      row.appendChild(timestampCell);
      row.appendChild(domainCell);
      row.appendChild(actionCell);
      row.appendChild(statusCell);
      
      // Add row to table
      activityTable.appendChild(row);
    });
    
    // Update pagination UI
    updatePagination(
      activityPagination,
      paginationState.activity.currentPage,
      paginationState.activity.totalPages,
      'activity'
    );
  }
  
  // Update NSFW sites table
  function updateNsfwSitesTable(sites) {
    // Filter to only show NSFW sites
    const nsfwSites = sites.filter(site => site.isNsfw);
    
    // Handle empty state
    if (!nsfwSites || nsfwSites.length === 0) {
      nsfwSitesTable.innerHTML = '';
      nsfwSitesEmptyState.style.display = 'block';
      nsfwPagination.innerHTML = '';
      return;
    }
    
    nsfwSitesEmptyState.style.display = 'none';
    
    // Set up pagination
    const totalItems = nsfwSites.length;
    paginationState.nsfwSites.totalPages = Math.ceil(totalItems / paginationState.nsfwSites.itemsPerPage);
    
    // Validate current page
    if (paginationState.nsfwSites.currentPage > paginationState.nsfwSites.totalPages) {
      paginationState.nsfwSites.currentPage = 1;
    }
    
    // Calculate slice indices
    const startIndex = (paginationState.nsfwSites.currentPage - 1) * paginationState.nsfwSites.itemsPerPage;
    const endIndex = Math.min(startIndex + paginationState.nsfwSites.itemsPerPage, totalItems);
    
    // Get page data
    const pageData = nsfwSites.slice(startIndex, endIndex);
    
    // Clear table
    nsfwSitesTable.innerHTML = '';
    
    // Populate table
    pageData.forEach(site => {
      const row = document.createElement('tr');
      
      // Create cells
      const domainCell = document.createElement('td');
      domainCell.textContent = site.domain;
      
      const firstVisitCell = document.createElement('td');
      firstVisitCell.textContent = site.firstVisit || 'Unknown';
      
      const lastVisitCell = document.createElement('td');
      lastVisitCell.textContent = site.lastVisit || 'Unknown';
      
      const visitsCell = document.createElement('td');
      visitsCell.textContent = site.visits || 1;
      
      const statusCell = document.createElement('td');
      const statusTag = document.createElement('span');
      statusTag.className = 'tag tag-nsfw';
      statusTag.textContent = 'NSFW';
      statusCell.appendChild(statusTag);
      
      // Add cells to row
      row.appendChild(domainCell);
      row.appendChild(firstVisitCell);
      row.appendChild(lastVisitCell);
      row.appendChild(visitsCell);
      row.appendChild(statusCell);
      
      // Add row to table
      nsfwSitesTable.appendChild(row);
    });
    
    // Update pagination UI
    updatePagination(
      nsfwPagination,
      paginationState.nsfwSites.currentPage,
      paginationState.nsfwSites.totalPages,
      'nsfwSites'
    );
  }
  
  // Update pagination UI
  function updatePagination(container, currentPage, totalPages, type) {
    container.innerHTML = '';
    
    if (totalPages <= 1) {
      return;
    }
    
    // Previous button
    const prevButton = document.createElement('button');
    prevButton.textContent = 'Previous';
    prevButton.disabled = currentPage === 1;
    prevButton.addEventListener('click', function() {
      if (currentPage > 1) {
        paginationState[type].currentPage--;
        if (type === 'activity') {
          chrome.storage.local.get(['nsfwUserActions'], function(result) {
            updateActivityTable(result.nsfwUserActions || []);
          });
        } else {
          chrome.storage.local.get(['nsfwSiteList'], function(result) {
            updateNsfwSitesTable(result.nsfwSiteList || []);
          });
        }
      }
    });
    container.appendChild(prevButton);
    
    // Page numbers - show up to 5 pages
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, startPage + 4);
    
    for (let i = startPage; i <= endPage; i++) {
      const pageButton = document.createElement('button');
      pageButton.textContent = i;
      pageButton.className = i === currentPage ? 'active' : '';
      pageButton.addEventListener('click', function() {
        paginationState[type].currentPage = i;
        if (type === 'activity') {
          chrome.storage.local.get(['nsfwUserActions'], function(result) {
            updateActivityTable(result.nsfwUserActions || []);
          });
        } else {
          chrome.storage.local.get(['nsfwSiteList'], function(result) {
            updateNsfwSitesTable(result.nsfwSiteList || []);
          });
        }
      });
      container.appendChild(pageButton);
    }
    
    // Next button
    const nextButton = document.createElement('button');
    nextButton.textContent = 'Next';
    nextButton.disabled = currentPage === totalPages;
    nextButton.addEventListener('click', function() {
      if (currentPage < totalPages) {
        paginationState[type].currentPage++;
        if (type === 'activity') {
          chrome.storage.local.get(['nsfwUserActions'], function(result) {
            updateActivityTable(result.nsfwUserActions || []);
          });
        } else {
          chrome.storage.local.get(['nsfwSiteList'], function(result) {
            updateNsfwSitesTable(result.nsfwSiteList || []);
          });
        }
      }
    });
    container.appendChild(nextButton);
  }
  
  // Format action type for display
  function formatActionType(action) {
    switch (action) {
      case 'warning_shown':
        return 'Warning Displayed';
      case 'proceed':
        return 'Proceeded to Site';
      default:
        return action.charAt(0).toUpperCase() + action.slice(1).replace('_', ' ');
    }
  }
  
  // Get status class for action
  function getActionStatusClass(action) {
    switch (action) {
      case 'warning_shown':
        return 'tag-warning';
      case 'proceed':
        return 'tag-nsfw';
      default:
        return 'tag-safe';
    }
  }
  
  // Get status text for action
  function getActionStatusText(action) {
    switch (action) {
      case 'warning_shown':
        return 'Warning';
      case 'proceed':
        return 'Accessed';
      default:
        return 'OK';
    }
  }
  
  // Clear all NSFW data
  function clearAllData() {
    chrome.storage.local.remove(['nsfwStats', 'nsfwSiteList', 'nsfwUserActions', 'nsfwUrlCache'], function() {
      // Reset UI
      totalSitesElem.textContent = '0';
      nsfwSitesElem.textContent = '0';
      detectionRateElem.textContent = '0.0%';
      blockedCountElem.textContent = '0';
      
      // Clear tables
      activityTable.innerHTML = '';
      activityEmptyState.style.display = 'block';
      activityPagination.innerHTML = '';
      
      nsfwSitesTable.innerHTML = '';
      nsfwSitesEmptyState.style.display = 'block';
      nsfwPagination.innerHTML = '';
      
      // Reset pagination
      paginationState.activity.currentPage = 1;
      paginationState.nsfwSites.currentPage = 1;
      
      // Show confirmation
      alert('All NSFW detection data has been cleared.');
    });
  }
  
  // Export data as JSON
  function exportData() {
    chrome.storage.local.get(['nsfwStats', 'nsfwSiteList', 'nsfwUserActions'], function(result) {
      // Create a formatted export object
      const exportData = {
        stats: result.nsfwStats || {},
        sites: result.nsfwSiteList || [],
        activity: result.nsfwUserActions || [],
        exportDate: new Date().toISOString(),
        extension: 'SecureGo',
        version: '1.0'
      };
      
      // Convert to JSON string
      const jsonString = JSON.stringify(exportData, null, 2);
      
      // Create blob
      const blob = new Blob([jsonString], {type: 'application/json'});
      
      // Create download link
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `securego_nsfw_export_${new Date().toISOString().split('T')[0]}.json`;
      
      // Trigger download
      document.body.appendChild(a);
      a.click();
      
      // Cleanup
      document.body.removeChild(a);
      window.URL.revokeObjectURL(a.href);
    });
  }
}); 