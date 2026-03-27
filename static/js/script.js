document.addEventListener('DOMContentLoaded', () => {
    // Hamburger Menu Logic
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('nav-links');

    if (hamburger) {
        hamburger.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            hamburger.classList.toggle('toggle');
        });
    }

    // Theme Toggle Logic
    const themeToggleBtn = document.getElementById('theme-toggle');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
            
            // Optional: You could update charts here if you want dynamic color changing.
        });
    }

    // Prediction Form Logic
    const predictForm = document.getElementById('predict-form');
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submit-btn');
            const btnText = submitBtn.querySelector('.btn-text');
            const loader = submitBtn.querySelector('.loader');
            const resultBox = document.getElementById('result-box');
            
            // Show loading
            btnText.style.display = 'none';
            loader.classList.remove('hidden');
            resultBox.classList.add('hidden');
            
            // Gather data
            const formData = new FormData(predictForm);
            const data = {};
            for (let [key, value] of formData.entries()) {
                const allVals = formData.getAll(key);
                data[key] = allVals.length > 1 ? allVals : value;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Update UI
                    document.getElementById('severity-level').innerText = `Level ${result.severity_level}`;
                    document.getElementById('severity-label').innerText = result.label;
                    document.getElementById('severity-desc').innerText = result.explanation;
                    
                    // Update badge color based on severity
                    const badge = document.getElementById('severity-badge');
                    let color = 'var(--success)';
                    if (result.severity_level === 2) color = '#3b82f6';
                    if (result.severity_level === 3) color = 'var(--warning)';
                    if (result.severity_level === 4) color = 'var(--danger)';
                    badge.style.borderColor = color;
                    badge.style.color = color;
                    
                    resultBox.classList.remove('hidden');
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (err) {
                alert('An error occurred during prediction.');
                console.error(err);
            } finally {
                // Restore button
                btnText.style.display = 'inline-block';
                loader.classList.add('hidden');
            }
        });
    }
});

// Chart.js Setup for Dashboard
function initCharts() {
    const ctxSeverity = document.getElementById('severityChart');
    const ctxFactors = document.getElementById('factorsChart');

    if (ctxSeverity) {
        new Chart(ctxSeverity, {
            type: 'doughnut',
            data: {
                labels: ['Level 1 (Low)', 'Level 2 (Medium)', 'Level 3 (High)', 'Level 4 (Critical)'],
                datasets: [{
                    data: [68582, 5000, 78005, 15000], // Dummy stats for aesthetics
                    backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' }
                }
            }
        });
    }

    if (ctxFactors) {
        new Chart(ctxFactors, {
            type: 'bar',
            data: {
                labels: ['Traffic Signal', 'Junction', 'Precipitation', 'Visibility', 'Crossing'],
                datasets: [{
                    label: 'Impact Score (Dummy Data)',
                    data: [10258, 10258, 20594, 10253,14000],
                    backgroundColor: 'rgba(99, 102, 241, 0.7)',
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } }
            }
        });
    }
}
