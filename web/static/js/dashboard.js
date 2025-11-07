/**
 * Dashboard-specific JavaScript
 * Handles sensor history charts and real-time updates
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize sensor history chart
    initializeSensorChart();

    // Enable auto-refresh (optional - can be toggled by user)
    // const autoRefreshInterval = enableAutoRefresh(60000); // 60 seconds
});

/**
 * Initialize sensor history chart
 */
function initializeSensorChart() {
    const chartCanvas = document.getElementById('sensorChart');

    if (!chartCanvas) {
        console.log('Sensor chart canvas not found');
        return;
    }

    // Get sensor data from hidden script tag
    const sensorDataElement = document.getElementById('sensor-data');
    let sensorData = [];

    if (sensorDataElement) {
        try {
            sensorData = JSON.parse(sensorDataElement.textContent);
        } catch (error) {
            console.error('Error parsing sensor data:', error);
            return;
        }
    }

    if (sensorData.length === 0) {
        console.log('No sensor data available for chart');
        return;
    }

    // Prepare data for chart
    const labels = sensorData.map(d => {
        const date = new Date(d.timestamp);
        return date.toLocaleTimeString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    });

    const temperatures = sensorData.map(d => d.temperature);
    const humidities = sensorData.map(d => d.humidity);
    const soilMoistures = sensorData.map(d => d.soil_moisture);

    // Create chart
    const ctx = chartCanvas.getContext('2d');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: temperatures,
                    borderColor: 'rgb(220, 53, 69)',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Humidity (%)',
                    data: humidities,
                    borderColor: 'rgb(13, 110, 253)',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y1'
                },
                {
                    label: 'Soil Moisture (%)',
                    data: soilMoistures,
                    borderColor: 'rgb(25, 135, 84)',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += context.parsed.y.toFixed(1);
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45,
                        maxTicksLimit: 12 // Limit number of x-axis labels
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Temperature (°C)'
                    },
                    min: 0,
                    max: 50
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Percentage (%)'
                    },
                    min: 0,
                    max: 100,
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });

    console.log('Sensor chart initialized with', sensorData.length, 'data points');
}

/**
 * Update chart with new data (for real-time updates)
 */
async function updateSensorChart() {
    try {
        const data = await window.dashboardAPI.fetchAPI('/api/sensor-data?hours=24');

        if (data.count > 0) {
            // Reinitialize chart with new data
            const sensorDataElement = document.getElementById('sensor-data');
            if (sensorDataElement) {
                sensorDataElement.textContent = JSON.stringify(data.data);
                initializeSensorChart();
            }
        }
    } catch (error) {
        console.error('Error updating sensor chart:', error);
    }
}
