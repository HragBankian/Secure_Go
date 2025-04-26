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

            if (!emailText.trim()) {
                showResult('Please enter email text to analyze', 'warning');
                return;
            }

            // Disable button and show loading state
            submitButton.disabled = true;
            submitButton.textContent = 'Analyzing...';

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
                    const message = `Classification: <strong>${data.result.toUpperCase()}</strong><br>
                                    Original Label: ${data.original_label}`;
                    showResult(message, resultType);
                } else {
                    showResult(`Error: ${data.error || 'Unknown error'}`, 'error');
                }
            } catch (error) {
                showResult(`Error: ${error.message}`, 'error');
            } finally {
                // Re-enable button
                submitButton.disabled = false;
                submitButton.textContent = 'Check Email';
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

            if (response.ok && data.status === 'ok') {
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
        }
    }

    // Initialize dashboard components if they exist
    initializeDashboardCharts();
});

// Placeholder function for future dashboard charts
function initializeDashboardCharts() {
    // This will be implemented later when dashboard features are added
    console.log('Dashboard ready for future integration');
} 