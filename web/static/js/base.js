// ===== BASE JAVASCRIPT =====

// Global variables
let isSystemOnline = true;
let currentTime = new Date();
let systemMetrics = {
    cpu: 23,
    memory: 45,
    network: 'Active'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Main initialization function
function initializeApp() {
    hideLoadingScreen();
    initializeNavigation();
    initializeStatusBar();
    initializeNotifications();
    initializeModals();
    startBackgroundAnimations();
    updateSystemMetrics();
}

// Loading screen management
function hideLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
        setTimeout(() => {
            loadingScreen.style.opacity = '0';
            setTimeout(() => {
                loadingScreen.style.display = 'none';
            }, 500);
        }, 2000);
    }
}

// Navigation functionality
function initializeNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const currentPage = window.location.pathname;

    navLinks.forEach(link => {
        const linkPage = link.getAttribute('data-page');
        if (currentPage.includes(linkPage) || (currentPage === '/' && linkPage === 'dashboard')) {
            link.classList.add('active');
        }

        link.addEventListener('click', function(e) {
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // User menu functionality
    const userAvatar = document.querySelector('.user-avatar');
    const userMenu = document.querySelector('.user-menu');
    
    if (userAvatar && userMenu) {
        userAvatar.addEventListener('click', function(e) {
            e.stopPropagation();
            userMenu.style.opacity = userMenu.style.opacity === '1' ? '0' : '1';
            userMenu.style.visibility = userMenu.style.visibility === 'visible' ? 'hidden' : 'visible';
            userMenu.style.transform = userMenu.style.transform === 'translateY(0px)' ? 'translateY(-10px)' : 'translateY(0px)';
        });

        document.addEventListener('click', function() {
            userMenu.style.opacity = '0';
            userMenu.style.visibility = 'hidden';
            userMenu.style.transform = 'translateY(-10px)';
        });
    }
}

// Status bar functionality
function initializeStatusBar() {
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);
    setInterval(updateSystemMetrics, 5000);
}

function updateCurrentTime() {
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        timeElement.textContent = timeString;
    }
}

function updateSystemMetrics() {
    // Simulate real-time metrics updates
    const cpuElement = document.getElementById('cpu-usage');
    const memoryElement = document.getElementById('memory-usage');
    const networkElement = document.getElementById('network-status');

    if (cpuElement) {
        systemMetrics.cpu = Math.max(10, Math.min(90, systemMetrics.cpu + (Math.random() - 0.5) * 10));
        cpuElement.textContent = `CPU: ${Math.round(systemMetrics.cpu)}%`;
    }

    if (memoryElement) {
        systemMetrics.memory = Math.max(20, Math.min(85, systemMetrics.memory + (Math.random() - 0.5) * 5));
        memoryElement.textContent = `RAM: ${Math.round(systemMetrics.memory)}%`;
    }

    if (networkElement) {
        const statuses = ['Active', 'Stable', 'Optimal'];
        systemMetrics.network = statuses[Math.floor(Math.random() * statuses.length)];
        networkElement.textContent = `Network: ${systemMetrics.network}`;
    }
}

// Notification system
function initializeNotifications() {
    window.showNotification = function(message, type = 'info', duration = 5000) {
        const container = document.getElementById('notification-container');
        if (!container) return;

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-message">${message}</div>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        container.appendChild(notification);

        // Auto-remove notification
        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => {
                    if (notification.parentElement) {
                        notification.remove();
                    }
                }, 300);
            }
        }, duration);
    };
}

// Modal system
function initializeModals() {
    window.showModal = function(content, title = '') {
        const modalContainer = document.getElementById('modal-container');
        if (!modalContainer) return;

        modalContainer.innerHTML = `
            <div class="modal active">
                <div class="modal-content">
                    ${title ? `<div class="modal-header">
                        <h3>${title}</h3>
                        <button class="modal-close" onclick="closeModal()">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>` : ''}
                    <div class="modal-body">
                        ${content}
                    </div>
                </div>
            </div>
        `;

        modalContainer.classList.add('active');
    };

    window.closeModal = function() {
        const modalContainer = document.getElementById('modal-container');
        if (modalContainer) {
            modalContainer.classList.remove('active');
            setTimeout(() => {
                modalContainer.innerHTML = '';
            }, 300);
        }
    };

    // Close modal on background click
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal-container')) {
            closeModal();
        }
    });
}

// Background animations
function startBackgroundAnimations() {
    // Particle effects
    createParticles();
    
    // Grid animation
    animateGrid();
    
    // Energy field pulse
    animateEnergyField();
}

function createParticles() {
    const particlesContainer = document.querySelector('.particles');
    if (!particlesContainer) return;

    for (let i = 0; i < 20; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            position: absolute;
            width: 2px;
            height: 2px;
            background: var(--primary-color);
            border-radius: 50%;
            opacity: ${Math.random() * 0.5 + 0.1};
            animation: particleFloat ${Math.random() * 10 + 10}s linear infinite;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
        `;
        particlesContainer.appendChild(particle);
    }
}

function animateGrid() {
    const grid = document.querySelector('.grid-overlay');
    if (grid) {
        let offset = 0;
        setInterval(() => {
            offset += 0.5;
            grid.style.transform = `translate(${offset}px, ${offset}px)`;
        }, 100);
    }
}

function animateEnergyField() {
    const energyField = document.querySelector('.energy-field');
    if (energyField) {
        let pulse = 0;
        setInterval(() => {
            pulse += 0.1;
            energyField.style.opacity = 0.3 + Math.sin(pulse) * 0.3;
        }, 100);
    }
}

// Utility functions
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('Application error:', e.error);
    showNotification('An error occurred. Please refresh the page.', 'error');
});

// Performance monitoring
function monitorPerformance() {
    if ('performance' in window) {
        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.entryType === 'navigation') {
                    console.log('Page load time:', entry.loadEventEnd - entry.loadEventStart);
                }
            }
        });
        observer.observe({ entryTypes: ['navigation'] });
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K: Focus search (if available)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[type="search"], .search-input');
        if (searchInput) {
            searchInput.focus();
        }
    }

    // Escape: Close modals
    if (e.key === 'Escape') {
        closeModal();
    }

    // Ctrl/Cmd + Shift + R: Refresh dashboard
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'R') {
        e.preventDefault();
        if (typeof refreshDashboard === 'function') {
            refreshDashboard();
        }
    }
});

// Service Worker registration (for PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}

// Export functions for use in other modules
window.AIvance = {
    showNotification,
    showModal,
    closeModal,
    formatNumber,
    formatBytes,
    debounce,
    throttle
}; 