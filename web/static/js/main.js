/**
 * Main JavaScript for Smart Irrigation Dashboard
 * Handles real-time updates, API calls, and UI interactions
 */

// Global configuration
const CONFIG = {
    refreshInterval: 60000, // 60 seconds
    apiEndpoints: {
        sensorData: '/api/sensor-data',
        irrigationDecision: '/api/irrigation-decision',
        readSensors: '/api/read-sensors',
        stats: '/api/stats'
    }
};

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    // Update live time
    updateLiveTime();
    setInterval(updateLiveTime, 1000);

    // Initialize tooltips (if Bootstrap tooltips are used)
    initializeTooltips();

    console.log('Smart Irrigation Dashboard initialized');
});

/**
 * Update live time display in footer
 */
function updateLiveTime() {
    const timeElement = document.getElementById('live-time');
    if (timeElement) {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        const dateString = now.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        timeElement.textContent = `${dateString} ${timeString}`;
    }
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(
        document.querySelectorAll('[data-bs-toggle="tooltip"]')
    );
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Show loading spinner
 */
function showLoading(message = 'Loading...') {
    let overlay = document.querySelector('.spinner-overlay');

    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'spinner-overlay';
        overlay.innerHTML = `
            <div class="spinner-content">
                <div class="spinner-border text-light" style="width: 3rem; height: 3rem;" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3 loading-message">${message}</p>
            </div>
        `;
        document.body.appendChild(overlay);
    } else {
        overlay.querySelector('.loading-message').textContent = message;
    }

    overlay.classList.add('active');
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    const overlay = document.querySelector('.spinner-overlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

/**
 * Fetch data from API
 */
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API fetch error:', error);
        showAlert('Error fetching data. Please try again.', 'error');
        throw error;
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'success') {
    const alertContainer = document.querySelector('.container');

    if (!alertContainer) return;

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert at top of container
    alertContainer.insertBefore(alertDiv, alertContainer.firstChild);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 150);
    }, 5000);
}

/**
 * Refresh dashboard data
 */
async function refreshDashboard() {
    showLoading('Refreshing dashboard...');

    try {
        // Trigger sensor reading
        await fetchAPI(CONFIG.apiEndpoints.readSensors, { method: 'POST' });

        // Reload page to show updated data
        setTimeout(() => {
            window.location.reload();
        }, 1000);

    } catch (error) {
        hideLoading();
        console.error('Refresh error:', error);
    }
}

/**
 * Get new irrigation recommendation
 */
async function getNewRecommendation() {
    showLoading('Analyzing sensor data...');

    try {
        // First, read sensors
        await fetchAPI(CONFIG.apiEndpoints.readSensors, { method: 'POST' });

        // Then get recommendation
        const result = await fetchAPI(CONFIG.apiEndpoints.irrigationDecision);

        hideLoading();

        if (result.error) {
            showAlert(`Error: ${result.error}`, 'error');
        } else {
            // Show result in a modal or reload page
            const decision = result.decision.replace(/_/g, ' ');
            const confidence = (result.confidence * 100).toFixed(1);

            let message = `<strong>${decision}</strong><br>`;
            if (result.duration > 0) {
                message += `Duration: ${result.duration} minutes<br>`;
            }
            message += `Confidence: ${confidence}%`;

            showAlert(message, 'success');

            // Reload after 2 seconds to show updated data
            setTimeout(() => window.location.reload(), 2000);
        }

    } catch (error) {
        hideLoading();
        console.error('Recommendation error:', error);
    }
}

/**
 * Format timestamp
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Create chart gradient
 */
function createGradient(ctx, color1, color2) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

/**
 * Auto-refresh functionality
 */
function enableAutoRefresh(intervalMs = CONFIG.refreshInterval) {
    return setInterval(() => {
        console.log('Auto-refreshing sensor data...');
        fetchAPI(CONFIG.apiEndpoints.readSensors, { method: 'POST' })
            .then(() => {
                // Update UI elements without full page reload
                updateSensorDisplay();
            })
            .catch(error => console.error('Auto-refresh error:', error));
    }, intervalMs);
}

/**
 * Update sensor display (AJAX update without page reload)
 */
async function updateSensorDisplay() {
    try {
        const data = await fetchAPI(CONFIG.apiEndpoints.sensorData + '?latest=true');

        // Update sensor values on page
        const tempElement = document.querySelector('[data-sensor="temperature"]');
        const humidityElement = document.querySelector('[data-sensor="humidity"]');
        const moistureElement = document.querySelector('[data-sensor="soil_moisture"]');

        if (tempElement) tempElement.textContent = `${data.temperature.toFixed(1)}°C`;
        if (humidityElement) humidityElement.textContent = `${data.humidity.toFixed(1)}%`;
        if (moistureElement) moistureElement.textContent = `${data.soil_moisture.toFixed(1)}%`;

        console.log('Sensor display updated');
    } catch (error) {
        console.error('Update display error:', error);
    }
}

// Export functions for use in other scripts
window.dashboardAPI = {
    refreshDashboard,
    getNewRecommendation,
    fetchAPI,
    showAlert,
    showLoading,
    hideLoading,
    formatTimestamp,
    createGradient
};
