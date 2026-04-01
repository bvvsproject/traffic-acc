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

    // Prediction Form Logic is handled natively inline within predict.html.
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
