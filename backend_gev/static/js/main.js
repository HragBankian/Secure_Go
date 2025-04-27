// Main JavaScript for Phishing Detection Dashboard

document.addEventListener('DOMContentLoaded', function () {
    // Form submission handler
    const emailForm = document.getElementById('email-check-form');
    if (emailForm) {
        emailForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            const emailText = document.getElementById('email-text').value;
            const resultDiv = document.getElementById('result');
            const submitButton = document.querySelector('#email-check-form button[type="submit"]');
            const originalButtonText = submitButton.innerHTML;

            if (!emailText.trim()) {
                showResult('Please enter email content to analyze', 'warning');
                return;
            }

            // Disable button and show loading state
            submitButton.disabled = true;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

            try {
                const response = await fetch('/AI', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ email: emailText })
                });

                const data = await response.json();

                if (response.ok) {
                    // Format and display the result
                    const resultType = data.result === 'spam' ? 'error' : 'success';
                    let message = '';
                    
                    if (resultType === 'error') {
                        message = `
                            <div class="result-header">
                                <i class="fas fa-exclamation-triangle"></i>
                                <h3>Phishing Detected</h3>
                            </div>
                            <p>This email appears to be a phishing attempt. We recommend not interacting with it.</p>
                            <div class="result-details">
                                <p><strong>Classification:</strong> ${data.result.toUpperCase()}</p>
                                <p><strong>Original Label:</strong> ${data.original_label || 'N/A'}</p>
                            </div>
                        `;
                    } else {
                        message = `
                            <div class="result-header">
                                <i class="fas fa-check-circle"></i>
                                <h3>Safe Email</h3>
                            </div>
                            <p>This email appears to be legitimate. However, always remain vigilant.</p>
                            <div class="result-details">
                                <p><strong>Classification:</strong> ${data.result.toUpperCase()}</p>
                                <p><strong>Original Label:</strong> ${data.original_label || 'N/A'}</p>
                            </div>
                        `;
                    }
                    
                    showResult(message, resultType);
                } else {
                    showResult(`
                        <div class="result-header">
                            <i class="fas fa-times-circle"></i>
                            <h3>Error</h3>
                        </div>
                        <p>${data.error || 'An unknown error occurred while analyzing the email.'}</p>
                    `, 'error');
                }
            } catch (error) {
                showResult(`
                    <div class="result-header">
                        <i class="fas fa-times-circle"></i>
                        <h3>Error</h3>
                    </div>
                    <p>${error.message || 'An unexpected error occurred. Please try again.'}</p>
                `, 'error');
            } finally {
                // Re-enable button
                submitButton.disabled = false;
                submitButton.innerHTML = originalButtonText;
            }
        });
    }

    // Function to display results
    function showResult(message, type) {
        const resultDiv = document.getElementById('result');
        if (resultDiv) {
            resultDiv.innerHTML = message;
            resultDiv.className = `result ${type}`;
            resultDiv.style.display = 'block';
            
            // Scroll to result
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    // Check API health on page load
    checkApiHealth();

    async function checkApiHealth() {
        const statusIndicator = document.getElementById('api-status');
        if (!statusIndicator) return;

        try {
            const response = await fetch('/health');
            const data = await response.json();

            if (response.ok && (data.status === 'ok' || data.status === 'healthy' || data.status === 'degraded')) {
                if (data.model_loaded) {
                    statusIndicator.textContent = 'Online';
                    statusIndicator.className = 'status-badge online';
                } else {
                    statusIndicator.textContent = 'Model Not Loaded';
                    statusIndicator.className = 'status-badge warning';
                }
            } else {
                statusIndicator.textContent = 'Error';
                statusIndicator.className = 'status-badge offline';
            }
        } catch (error) {
            statusIndicator.textContent = 'Offline';
            statusIndicator.className = 'status-badge offline';
            console.error('API health check failed:', error);
        }
    }

    // Add smooth scrolling for all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // If we're on the homepage, add some animations
    if (document.querySelector('.hero')) {
        const featureCards = document.querySelectorAll('.feature-card');
        
        // Simple animation on scroll
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        
        featureCards.forEach(card => {
            observer.observe(card);
        });
    }

    // Initialize dashboard components if they exist
    initializeDashboardCharts();
});

// Placeholder function for future dashboard charts
function initializeDashboardCharts() {
    // This will be implemented later when dashboard features are added
    console.log('Dashboard ready for future integration');
}

// Add CSS for animate-in class
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
        .feature-card {
            opacity: 0;
            transform: translateY(20px);
            transition: opacity 0.5s ease, transform 0.5s ease;
        }
        
        .feature-card.animate-in {
            opacity: 1;
            transform: translateY(0);
        }
        
        .result-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .result-header i {
            font-size: 1.8rem;
            margin-right: 10px;
        }
        
        .result-header h3 {
            margin: 0;
        }
        
        .result-details {
            background-color: rgba(0, 0, 0, 0.05);
            padding: 10px 15px;
            border-radius: 4px;
            margin-top: 15px;
        }
        
        .result-details p {
            margin: 5px 0;
        }
    `;
    document.head.appendChild(style);
}); 