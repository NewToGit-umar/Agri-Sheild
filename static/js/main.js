/**
 * AgriShield AI — Client-Side Controller
 * Handles tab switching, file upload, disease analysis, and chatbot interactions.
 */

// ═══════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════
const state = {
    currentTab: 'scanner',
    selectedFile: null,
    isAnalyzing: false,
    isChatting: false,
    scannerMode: 'file', // 'file' or 'camera'
    cameraStream: null,
    cameraDevices: [],
    currentCameraIdx: 0,
    user: null,
    authMode: 'login', // 'login' or 'signup'
};

// ═══════════════════════════════════════════════════════════════════════════
// DOM REFERENCES
// ═══════════════════════════════════════════════════════════════════════════
const dom = {
    // Tabs
    tabScanner:       () => document.getElementById('tab-scanner'),
    tabChatbot:       () => document.getElementById('tab-chatbot'),
    tabDashboard:     () => document.getElementById('tab-dashboard'),
    panelScanner:     () => document.getElementById('panel-scanner'),
    panelChatbot:     () => document.getElementById('panel-chatbot'),
    panelDashboard:   () => document.getElementById('panel-dashboard'),

    // Scanner
    dropZone:         () => document.getElementById('drop-zone'),
    fileInput:        () => document.getElementById('file-input'),
    uploadPlaceholder:() => document.getElementById('upload-placeholder'),
    previewContainer: () => document.getElementById('image-preview-container'),
    imagePreview:     () => document.getElementById('image-preview'),
    fileName:         () => document.getElementById('file-name'),
    analyzeBtn:       () => document.getElementById('analyze-btn'),
    analyzeBtnText:   () => document.getElementById('analyze-btn-text'),
    analyzeSpinner:   () => document.getElementById('analyze-spinner'),
    resultsEmpty:     () => document.getElementById('results-empty'),
    resultsLoading:   () => document.getElementById('results-loading'),
    resultsContent:   () => document.getElementById('results-content'),

    // Camera
    cameraContainer:  () => document.getElementById('camera-container'),
    cameraVideo:      () => document.getElementById('camera-video'),
    cameraCanvas:     () => document.getElementById('camera-canvas'),
    cameraLoading:    () => document.getElementById('camera-loading'),
    captureBtn:       () => document.getElementById('capture-btn'),
    switchCameraBtn:  () => document.getElementById('switch-camera-btn'),
    toggleFileMode:   () => document.getElementById('toggle-file-mode'),
    toggleCamMode:    () => document.getElementById('toggle-cam-mode'),

    // Chat
    chatMessages:     () => document.getElementById('chat-messages'),
    chatInput:        () => document.getElementById('chat-input'),
    chatForm:         () => document.getElementById('chat-form'),
    sendBtn:          () => document.getElementById('send-btn'),
    sendBtnText:      () => document.getElementById('send-btn-text'),
    sendSpinner:      () => document.getElementById('send-spinner'),
    typingIndicator:  () => document.getElementById('typing-indicator'),

    // Auth
    authOverlay:      () => document.getElementById('auth-overlay'),
    authForm:         () => document.getElementById('auth-form'),
    authUsername:     () => document.getElementById('auth-username'),
    authEmail:        () => document.getElementById('auth-email'),
    authPassword:     () => document.getElementById('auth-password'),
    authError:        () => document.getElementById('auth-error'),
    authBtnText:      () => document.getElementById('auth-btn-text'),
    authSpinner:      () => document.getElementById('auth-spinner'),
    authTabLogin:     () => document.getElementById('auth-tab-login'),
    authTabSignup:    () => document.getElementById('auth-tab-signup'),
    emailGroup:       () => document.getElementById('email-group'),
    usernameLabel:    () => document.getElementById('username-label'),
    auth2FAContainer: () => document.getElementById('auth-2fa-container'),
    authCredentialsContainer: () => document.getElementById('auth-credentials-container'),
    
    userBadge:        () => document.getElementById('user-profile-badge'),
    userGreeting:     () => document.getElementById('user-greeting-name'),
    navSignoutBtn:    () => document.getElementById('nav-signout-btn'),
};

// ═══════════════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════════════════════════════════════
function switchTab(tab) {
    state.currentTab = tab;

    // Stop active camera stream when leaving scanner
    if (tab !== 'scanner') {
        stopCamera();
    }

    const scannerTab   = dom.tabScanner();
    const chatbotTab   = dom.tabChatbot();
    const dashboardTab = dom.tabDashboard();
    
    const scannerPanel   = dom.panelScanner();
    const chatbotPanel   = dom.panelChatbot();
    const dashboardPanel = dom.panelDashboard();

    const activeClasses   = 'text-white bg-emerald-700 shadow-md';
    const inactiveClasses = 'text-slate-600 hover:text-emerald-700';

    // Set classes to inactive
    scannerTab.className   = `tab-btn relative px-6 py-2.5 text-sm font-semibold rounded-lg transition-all duration-300 ${inactiveClasses}`;
    chatbotTab.className   = `tab-btn relative px-6 py-2.5 text-sm font-semibold rounded-lg transition-all duration-300 ${inactiveClasses}`;
    dashboardTab.className = `tab-btn relative px-6 py-2.5 text-sm font-semibold rounded-lg transition-all duration-300 ${inactiveClasses}`;

    // Hide all panels
    scannerPanel.classList.remove('active');
    chatbotPanel.classList.remove('active');
    dashboardPanel.classList.remove('active');

    if (tab === 'scanner') {
        scannerTab.className = `tab-btn relative px-6 py-2.5 text-sm font-semibold rounded-lg transition-all duration-300 ${activeClasses}`;
        scannerPanel.classList.add('active');
        
        // Re-initialize camera if we switch back to scanner and camera mode was active
        if (state.scannerMode === 'camera') {
            initCamera();
        }
    } else if (tab === 'chatbot') {
        chatbotTab.className = `tab-btn relative px-6 py-2.5 text-sm font-semibold rounded-lg transition-all duration-300 ${activeClasses}`;
        chatbotPanel.classList.add('active');
        scrollChatToBottom();
    } else if (tab === 'dashboard') {
        dashboardTab.className = `tab-btn relative px-6 py-2.5 text-sm font-semibold rounded-lg transition-all duration-300 ${activeClasses}`;
        dashboardPanel.classList.add('active');
        // Render scan history and dashboard widgets
        renderDashboardData();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DRAG & DROP + FILE INPUT
// ═══════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    // Check authentication on startup
    checkAuthStatus();
    
    // Initialize 2FA digits navigation
    setupOTPInputNavigation();

    const dropZone  = dom.dropZone();
    const fileInput = dom.fileInput();

    // Drag events
    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    // Standard file input
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });
});

function handleFileSelection(file) {
    const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];

    if (!allowed.includes(file.type)) {
        showError('results', 'Unsupported file format. Please upload a PNG, JPG, JPEG, or WEBP image.');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError('results', 'File exceeds the 16MB size limit.');
        return;
    }

    state.selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        dom.imagePreview().src = e.target.result;
        dom.fileName().textContent = file.name;
        dom.uploadPlaceholder().classList.add('hidden');
        dom.previewContainer().classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Enable button
    dom.analyzeBtn().disabled = false;

    // Reset results to empty state
    dom.resultsEmpty().classList.remove('hidden');
    dom.resultsLoading().classList.add('hidden');
    dom.resultsContent().classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════════════════════
// DISEASE ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════
async function analyzeCrop() {
    if (state.isAnalyzing || !state.selectedFile) return;

    state.isAnalyzing = true;
    lockButton('analyze');

    // Show loading state in results
    dom.resultsEmpty().classList.add('hidden');
    dom.resultsContent().classList.add('hidden');
    dom.resultsLoading().classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('image', state.selectedFile);

        const scanMode = document.getElementById('scan-mode').value;
        const endpoint = scanMode === 'fruit' ? '/api/analyze_fruit' : '/api/analyze';

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            renderDiagnosis(data.data);
        } else {
            showError('results', data.error || 'Analysis failed. Please try again.');
        }
    } catch (err) {
        showError('results', 'Network error: Unable to reach the analysis server. Please check your connection.');
    } finally {
        state.isAnalyzing = false;
        unlockButton('analyze');
    }
}

function renderDiagnosis(d) {
    const container = dom.resultsContent();

    // Determine status styling
    const statusMap = {
        'Healthy':             { color: 'emerald', icon: '✅', bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-800' },
        'Diseased':            { color: 'red',     icon: '🔴', bg: 'bg-red-50',     border: 'border-red-200',     text: 'text-red-800' },
        'Pest Infestation':    { color: 'orange',  icon: '🐛', bg: 'bg-orange-50',  border: 'border-orange-200',  text: 'text-orange-800' },
        'Nutrient Deficiency': { color: 'amber',   icon: '⚠️', bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-800' },
        'Unknown':             { color: 'slate',   icon: '❓', bg: 'bg-slate-50',   border: 'border-slate-200',   text: 'text-slate-700' },
    };

    const status = statusMap[d.health_status] || statusMap['Unknown'];

    // Parse confidence value for the bar
    const confidenceNum = parseInt(d.confidence) || 0;

    container.innerHTML = `
        <!-- Status Banner -->
        <div class="${status.bg} ${status.border} border rounded-xl p-4 animate-slide-up">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                    <span class="text-xl">${status.icon}</span>
                    <span class="text-sm font-bold ${status.text}">${d.health_status}</span>
                </div>
                <span class="text-xs font-medium text-slate-500 bg-white px-2.5 py-1 rounded-full border border-slate-100">
                    ${d.plant_type}
                </span>
            </div>
        </div>

        <!-- Diagnosis -->
        <div class="bg-white rounded-xl border border-slate-100 p-4 animate-slide-up" style="animation-delay: 0.1s;">
            <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Diagnosis</h4>
            <p class="text-sm font-semibold text-slate-800">${escapeHtml(d.diagnosis)}</p>
        </div>

        <!-- Confidence Meter -->
        <div class="bg-white rounded-xl border border-slate-100 p-4 animate-slide-up" style="animation-delay: 0.15s;">
            <div class="flex items-center justify-between mb-2">
                <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Confidence</h4>
                <span class="text-sm font-bold ${status.text}">${d.confidence}</span>
            </div>
            <div class="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                <div class="confidence-bar h-full rounded-full bg-gradient-to-r from-emerald-400 to-emerald-600"
                     style="width: 0%;"
                     data-target="${confidenceNum}">
                </div>
            </div>
        </div>

        <!-- Symptoms -->
        ${d.symptoms && d.symptoms.length > 0 ? `
        <div class="bg-white rounded-xl border border-slate-100 p-4 animate-slide-up" style="animation-delay: 0.2s;">
            <h4 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Observed Symptoms</h4>
            <ul class="space-y-1.5">
                ${d.symptoms.map(s => `
                    <li class="flex items-start gap-2 text-sm text-slate-700">
                        <span class="w-1.5 h-1.5 rounded-full bg-red-400 mt-1.5 flex-shrink-0"></span>
                        ${escapeHtml(s)}
                    </li>
                `).join('')}
            </ul>
        </div>
        ` : ''}

        <!-- Medicine / Treatment -->
        ${d.medicine && d.medicine.length > 0 ? `
        <div class="bg-blue-50/60 rounded-xl border border-blue-100 p-4 animate-slide-up" style="animation-delay: 0.22s;">
            <h4 class="text-xs font-semibold text-blue-600 uppercase tracking-wider mb-2">Recommended Medicine / Treatment</h4>
            <ul class="space-y-2">
                ${d.medicine.map(m => `
                    <li class="flex items-start gap-2 text-sm text-blue-900">
                        <span class="text-blue-500 mt-0.5 flex-shrink-0">💊</span>
                        ${escapeHtml(m)}
                    </li>
                `).join('')}
            </ul>
        </div>
        ` : ''}

        <!-- Management Recommendations -->
        ${d.management && d.management.length > 0 ? `
        <div class="bg-emerald-50/60 rounded-xl border border-emerald-100 p-4 animate-slide-up" style="animation-delay: 0.25s;">
            <h4 class="text-xs font-semibold text-emerald-600 uppercase tracking-wider mb-2">Management Recommendations</h4>
            <ul class="space-y-2">
                ${d.management.map(m => `
                    <li class="flex items-start gap-2 text-sm text-emerald-900">
                        <span class="text-emerald-500 mt-0.5 flex-shrink-0">🌿</span>
                        ${escapeHtml(m)}
                    </li>
                `).join('')}
            </ul>
        </div>
        ` : ''}
    `;

    // Show results
    dom.resultsLoading().classList.add('hidden');
    dom.resultsContent().classList.remove('hidden');

    // Save scan to user journal
    saveScanToJournal(d);

    // Animate confidence bar
    requestAnimationFrame(() => {
        setTimeout(() => {
            const bar = container.querySelector('.confidence-bar');
            if (bar) {
                bar.style.width = bar.dataset.target + '%';
            }
        }, 100);
    });
}

function clearScan() {
    state.selectedFile = null;

    // Reset file input
    dom.fileInput().value = '';

    // Reset preview
    dom.uploadPlaceholder().classList.remove('hidden');
    dom.previewContainer().classList.add('hidden');
    dom.imagePreview().src = '';
    dom.fileName().textContent = '';

    // Disable button
    dom.analyzeBtn().disabled = true;

    // Reset results
    dom.resultsEmpty().classList.remove('hidden');
    dom.resultsLoading().classList.add('hidden');
    dom.resultsContent().classList.add('hidden');
    dom.resultsContent().innerHTML = '';
}

// ═══════════════════════════════════════════════════════════════════════════
// CAMERA LIFE CYCLE & CAPTURE
// ═══════════════════════════════════════════════════════════════════════════
function switchScannerMode(mode) {
    state.scannerMode = mode;

    const fileBtn = dom.toggleFileMode();
    const camBtn = dom.toggleCamMode();
    const dropZone = dom.dropZone();
    const camContainer = dom.cameraContainer();

    const activeClasses = 'bg-white text-emerald-800 shadow-sm border border-slate-200';
    const inactiveClasses = 'text-slate-600 hover:text-emerald-700';

    if (mode === 'file') {
        // Toggle buttons styling
        fileBtn.className = `flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${activeClasses}`;
        camBtn.className = `flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${inactiveClasses}`;

        // Toggle visibility
        dropZone.classList.remove('hidden');
        camContainer.classList.add('hidden');

        // Stop stream
        stopCamera();
    } else {
        // Toggle buttons styling
        camBtn.className = `flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${activeClasses}`;
        fileBtn.className = `flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${inactiveClasses}`;

        // Toggle visibility
        dropZone.classList.add('hidden');
        camContainer.classList.remove('hidden');

        // Initialize stream
        initCamera();
    }
}

async function initCamera() {
    stopCamera(); // Make sure previous streams are cleared

    const video = dom.cameraVideo();
    const loading = dom.cameraLoading();
    const switchBtn = dom.switchCameraBtn();

    // Check if camera API is supported/available (blocked in non-secure HTTP contexts on mobile)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        loading.classList.add('hidden');
        alert("📷 Camera Access Blocked (Non-Secure Context)\n\n" +
              "Modern web browsers block camera access unless the website is served over a secure connection.\n\n" +
              "To solve this:\n" +
              "1. On Desktop: Access the app via http://localhost:5000 or http://127.0.0.1:5000 (browsers exempt localhost from this rule).\n" +
              "2. On Mobile: Access the app using an HTTPS tunnel (e.g., ngrok) or configure HTTPS local certificates.");
        switchScannerMode('file');
        return;
    }

    loading.style.opacity = '1';
    loading.classList.remove('hidden');
    switchBtn.classList.add('hidden');

    const constraints = {
        video: {
            facingMode: 'environment', // Prefer rear camera on mobile
            width: { ideal: 1280 },
            height: { ideal: 720 }
        },
        audio: false
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        state.cameraStream = stream;
        video.srcObject = stream;
        
        // Wait for video to load metadata and begin playing
        video.onloadedmetadata = () => {
            loading.style.opacity = '0';
            setTimeout(() => loading.classList.add('hidden'), 300);
        };

        // Enumerate other camera inputs to support camera swapping
        await enumerateDevices();

    } catch (err) {
        console.error('Camera initialization failed:', err);
        loading.classList.add('hidden');
        
        let errorMsg = 'Could not access the camera. ';
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            errorMsg += 'Permission was denied. Please allow camera access in your browser settings.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            errorMsg += 'No camera device was found on this system.';
        } else {
            errorMsg += 'Please ensure you are using a secure connection (HTTPS or localhost) and try again.';
        }
        
        alert(errorMsg);
        switchScannerMode('file');
    }
}

function stopCamera() {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => {
            track.stop();
        });
        state.cameraStream = null;
    }
    
    const video = dom.cameraVideo();
    if (video) {
        video.srcObject = null;
    }
}

async function enumerateDevices() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        
        // Filter unique video inputs with a valid deviceId
        const uniqueDevices = [];
        const seenIds = new Set();
        for (const device of devices) {
            if (device.kind === 'videoinput' && device.deviceId && !seenIds.has(device.deviceId)) {
                seenIds.add(device.deviceId);
                uniqueDevices.push(device);
            }
        }
        state.cameraDevices = uniqueDevices;
        
        // Synchronize current index with the active stream's device ID
        if (state.cameraStream) {
            const tracks = state.cameraStream.getVideoTracks();
            if (tracks.length > 0) {
                const settings = tracks[0].getSettings();
                const activeDeviceId = settings.deviceId;
                if (activeDeviceId) {
                    const idx = state.cameraDevices.findIndex(d => d.deviceId === activeDeviceId);
                    if (idx !== -1) {
                        state.currentCameraIdx = idx;
                    }
                }
            }
        }
        
        const switchBtn = dom.switchCameraBtn();
        if (state.cameraDevices.length > 1) {
            switchBtn.classList.remove('hidden');
        } else {
            switchBtn.classList.add('hidden');
        }
    } catch (err) {
        console.warn('Could not enumerate camera devices:', err);
    }
}

async function switchCamera() {
    if (state.cameraDevices.length <= 1) return;

    // Toggle index
    state.currentCameraIdx = (state.currentCameraIdx + 1) % state.cameraDevices.length;
    const nextDevice = state.cameraDevices[state.currentCameraIdx];

    const video = dom.cameraVideo();
    const loading = dom.cameraLoading();

    loading.style.opacity = '1';
    loading.classList.remove('hidden');

    stopCamera();

    const constraints = {
        video: {
            deviceId: { exact: nextDevice.deviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 }
        },
        audio: false
    };

    try {
        let stream;
        try {
            // Attempt to get user media with optimal resolution constraints
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (resErr) {
            console.warn('Failed to switch camera with resolution constraints, trying fallback:', resErr);
            // Fallback: request camera without resolution constraints to prevent OverconstrainedError
            const fallbackConstraints = {
                video: {
                    deviceId: { exact: nextDevice.deviceId }
                },
                audio: false
            };
            stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        }

        state.cameraStream = stream;
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            loading.style.opacity = '0';
            setTimeout(() => loading.classList.add('hidden'), 300);
        };
    } catch (err) {
        console.error('Failed to switch camera device completely:', err);
        // Fallback to auto initialization if both attempts fail
        initCamera();
    }
}

function capturePhoto() {
    const video = dom.cameraVideo();
    const canvas = dom.cameraCanvas();

    if (!video || !state.cameraStream) {
        alert('Camera stream is not active.');
        return;
    }

    // Set canvas dimensions equal to current video display size/aspect
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    
    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas image to Blob
    canvas.toBlob((blob) => {
        if (!blob) {
            alert('Failed to capture frame from stream.');
            return;
        }

        // Generate file object
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `camera-capture-${timestamp}.jpg`;
        const file = new File([blob], filename, { type: 'image/jpeg' });

        // Handle selected file (updates previews & triggers state change)
        handleFileSelection(file);

        // Customize displayed file name for camera capture
        dom.fileName().textContent = `📷 Live Photo Capture (${filename.slice(0, 20)}...)`;

        // Stop camera and return to preview upload mode
        switchScannerMode('file');

    }, 'image/jpeg', 0.95);
}

// ═══════════════════════════════════════════════════════════════════════════
// CHATBOT
// ═══════════════════════════════════════════════════════════════════════════
async function sendMessage(e) {
    e.preventDefault();

    const input = dom.chatInput();
    const message = input.value.trim();

    if (!message || state.isChatting) return;

    state.isChatting = true;
    lockButton('send');

    // Add user bubble
    appendChatBubble('user', message);
    input.value = '';

    // Show typing indicator
    dom.typingIndicator().classList.remove('hidden');
    scrollChatToBottom();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        const data = await response.json();

        // Hide typing
        dom.typingIndicator().classList.add('hidden');

        if (data.success) {
            appendChatBubble('ai', data.reply);
        } else {
            appendChatBubble('ai', `⚠️ ${data.error || 'Something went wrong. Please try again.'}`);
        }
    } catch (err) {
        dom.typingIndicator().classList.add('hidden');
        appendChatBubble('ai', '⚠️ Network error: Unable to reach the chat server. Please check your connection.');
    } finally {
        state.isChatting = false;
        unlockButton('send');
        input.focus();
    }
}

function appendChatBubble(role, text) {
    const container = dom.chatMessages();
    const wrapper = document.createElement('div');
    wrapper.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;
    wrapper.style.opacity = '0';
    wrapper.style.transform = 'translateY(10px)';

    const bubble = document.createElement('div');
    bubble.className = role === 'user' ? 'chat-bubble-user px-4 py-3' : 'chat-bubble-ai px-4 py-3';

    // Format text with simple markdown-like rendering
    bubble.innerHTML = `<p class="text-sm leading-relaxed whitespace-pre-wrap">${formatChatText(text)}</p>`;

    wrapper.appendChild(bubble);
    container.appendChild(wrapper);

    // Animate in
    requestAnimationFrame(() => {
        wrapper.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        wrapper.style.opacity = '1';
        wrapper.style.transform = 'translateY(0)';
    });

    scrollChatToBottom();
}

function formatChatText(text) {
    // Escape HTML first
    let safe = escapeHtml(text);
    
    // Headers: ### Title
    safe = safe.replace(/###\s+(.+)/g, '<h5 class="font-bold text-emerald-950 text-sm mt-3 mb-1.5">$1</h5>');
    safe = safe.replace(/##\s+(.+)/g, '<h4 class="font-bold text-emerald-950 text-base mt-4 mb-2">$1</h4>');
    
    // Bold: **text**
    safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic: *text*
    safe = safe.replace(/\*(.+?)\*/g, '<em>$1</em>');
    
    // Bullet points (with green bullet accent)
    safe = safe.replace(/^[\-•]\s+(.+)$/gm, '<div class="flex items-start gap-2 my-1 text-slate-700"><span class="text-emerald-500 mt-0.5">•</span><span>$1</span></div>');
    
    return safe;
}

async function clearChat() {
    try {
        await fetch('/api/chat/clear', { method: 'POST' });
    } catch (err) {
        // Ignore network errors on clear
    }

    // Reset UI
    const container = dom.chatMessages();
    container.innerHTML = `
        <div class="flex justify-start">
            <div class="chat-bubble-ai px-4 py-3">
                <p class="text-sm leading-relaxed">
                    👋 <strong>Hello!</strong> I'm your AgriShield AI Advisor — a virtual senior agronomist here to help you with soil management, irrigation, pest control, crop rotation, and all things farming. How can I assist you today?
                </p>
            </div>
        </div>
    `;
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════
function scrollChatToBottom() {
    const messages = dom.chatMessages();
    requestAnimationFrame(() => {
        messages.scrollTop = messages.scrollHeight;
    });
}

function lockButton(type) {
    if (type === 'analyze') {
        dom.analyzeBtn().disabled = true;
        dom.analyzeBtnText().textContent = 'Analyzing…';
        dom.analyzeSpinner().classList.remove('hidden');
    } else if (type === 'send') {
        dom.sendBtn().disabled = true;
        dom.sendBtnText().textContent = 'Sending…';
        dom.sendSpinner().classList.remove('hidden');
    }
}

function unlockButton(type) {
    if (type === 'analyze') {
        dom.analyzeBtn().disabled = !state.selectedFile;
        dom.analyzeBtnText().textContent = 'Analyze Disease';
        dom.analyzeSpinner().classList.add('hidden');
    } else if (type === 'send') {
        dom.sendBtn().disabled = false;
        dom.sendBtnText().textContent = 'Send';
        dom.sendSpinner().classList.add('hidden');
    }
}

function showError(target, message) {
    if (target === 'results') {
        const container = dom.resultsContent();
        container.innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-xl p-4 animate-slide-up">
                <div class="flex items-start gap-3">
                    <svg class="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"/>
                    </svg>
                    <div>
                        <p class="text-sm font-semibold text-red-800">Analysis Error</p>
                        <p class="text-xs text-red-600 mt-1">${escapeHtml(message)}</p>
                    </div>
                </div>
            </div>
        `;
        dom.resultsLoading().classList.add('hidden');
        dom.resultsEmpty().classList.add('hidden');
        dom.resultsContent().classList.remove('hidden');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════════════════════
// NEW UPGRADES (SUGGESTION CHIPS & LIBRARY ACCORDIONS)
// ═══════════════════════════════════════════════════════════════════════════
function toggleLibrary(type) {
    const panel = document.getElementById(`lib-${type}`);
    const arrow = document.getElementById(`arrow-${type}`);
    
    if (panel && arrow) {
        const isHidden = panel.classList.toggle('hidden');
        if (!isHidden) {
            arrow.classList.add('rotate-180');
        } else {
            arrow.classList.remove('rotate-180');
        }
    }
}

function sendSuggestion(text) {
    const input = dom.chatInput();
    if (input) {
        input.value = text;
        const form = dom.chatForm();
        if (form) {
            // Trigger standard form submit
            const event = new Event('submit', { cancelable: true });
            form.dispatchEvent(event);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// USER AUTHENTICATION SYSTEM (SQLite + SESSION)
// ═══════════════════════════════════════════════════════════════════════════
async function checkAuthStatus() {
    try {
        const response = await fetch('/api/auth/status');
        const data = await response.json();
        
        if (data.authenticated) {
            state.user = data.user;
            setupAuthenticatedUI();
        } else {
            state.user = null;
            showAuthOverlay();
        }
    } catch (e) {
        console.error('Auth check failed:', e);
        // Fallback to overlay if check fails
        showAuthOverlay();
    }
}

function showAuthOverlay() {
    const overlay = dom.authOverlay();
    if (overlay) {
        overlay.classList.remove('hidden');
    }
    // Hide user nav elements
    dom.userBadge().classList.add('hidden');
    dom.navSignoutBtn().classList.add('hidden');
}

function setupAuthenticatedUI() {
    const overlay = dom.authOverlay();
    if (overlay) {
        overlay.classList.add('hidden');
    }
    
    // Display greeting and sign out
    const nameSpan = dom.userGreeting();
    if (nameSpan && state.user) {
        nameSpan.textContent = state.user.username;
    }
    
    dom.userBadge().classList.remove('hidden');
    dom.navSignoutBtn().classList.remove('hidden');
}

function switchAuthTab(mode) {
    state.authMode = mode;
    
    const loginTab  = dom.authTabLogin();
    const signupTab = dom.authTabSignup();
    const emailGroup = dom.emailGroup();
    const usernameLabel = dom.usernameLabel();
    const btnText = dom.authBtnText();
    const errorBox = dom.authError();

    errorBox.classList.add('hidden');
    errorBox.textContent = '';

    const activeClasses = 'bg-white text-emerald-800 shadow-sm border border-slate-200';
    const inactiveClasses = 'text-slate-600 hover:text-emerald-700';

    if (mode === 'login') {
        loginTab.className = `flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${activeClasses}`;
        signupTab.className = `flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${inactiveClasses}`;
        emailGroup.classList.add('hidden');
        dom.authEmail().required = false;
        usernameLabel.textContent = 'Username or Email';
        btnText.textContent = 'Sign In';
    } else {
        signupTab.className = `flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${activeClasses}`;
        loginTab.className = `flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${inactiveClasses}`;
        emailGroup.classList.remove('hidden');
        dom.authEmail().required = true;
        usernameLabel.textContent = 'Desired Username';
        btnText.textContent = 'Register';
    }
}

async function handleAuthSubmit(e) {
    e.preventDefault();
    
    const usernameInput = dom.authUsername();
    const emailInput = dom.authEmail();
    const passwordInput = dom.authPassword();
    const errorBox = dom.authError();
    const spinner = dom.authSpinner();
    const btnText = dom.authBtnText();
    
    errorBox.classList.add('hidden');
    errorBox.textContent = '';
    spinner.classList.remove('hidden');
    btnText.textContent = state.authMode === 'login' ? 'Signing In...' : 'Registering...';

    const payload = {
        username: usernameInput.value.trim(),
        password: passwordInput.value
    };
    
    if (state.authMode === 'signup') {
        payload.email = emailInput.value.trim();
    }
    
    const endpoint = state.authMode === 'login' ? '/api/login' : '/api/signup';
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.two_factor_required) {
                // Switch to 2FA PIN Overlay Screen
                const credentialsCont = dom.authCredentialsContainer();
                const container2FA = dom.auth2FAContainer();
                if (credentialsCont) credentialsCont.classList.add('hidden');
                if (container2FA) container2FA.classList.remove('hidden');
                
                const emailSpan = document.getElementById('auth-2fa-email');
                if (emailSpan) emailSpan.textContent = data.email;
                
                // Clear any existing OTP values and focus first digit input
                const inputs = document.querySelectorAll('#otp-inputs-container .otp-digit');
                inputs.forEach(inp => inp.value = '');
                setTimeout(() => { if (inputs[0]) inputs[0].focus(); }, 100);
                
                // Start 10 minute countdown timer
                start2FATimer(600);
            } else {
                state.user = data.user;
                setupAuthenticatedUI();
                
                // Welcome msg in chatbot
                if (state.authMode === 'signup') {
                    appendChatBubble('ai', `👋 **Welcome, ${state.user.username}!** Thank you for registering. I'm your AgriShield AI advisor. I can help you with leaf/fruit diagnoses, dosage estimations, soil moisture issues, and planting rotations. Try out our new **Smart Tools** tab for active field calculators!`);
                }
                
                // Clear fields
                usernameInput.value = '';
                emailInput.value = '';
                passwordInput.value = '';
            }
        } else {
            errorBox.textContent = data.error || 'Authentication failed. Please verify fields.';
            errorBox.classList.remove('hidden');
        }
    } catch (err) {
        errorBox.textContent = 'Network error: Server authentication service unreachable.';
        errorBox.classList.remove('hidden');
    } finally {
        spinner.classList.add('hidden');
        btnText.textContent = state.authMode === 'login' ? 'Sign In' : 'Register';
    }
}

async function handleSignOut() {
    try {
        await fetch('/api/logout', { method: 'POST' });
    } catch (e) {}
    
    state.user = null;
    showAuthOverlay();
    
    // Clear chat UI and scanner selection
    clearChat();
    clearScan();
}

// ═══════════════════════════════════════════════════════════════════════════
// SCAN HISTORY JOURNAL (LOCAL PERSISTENCE)
// ═══════════════════════════════════════════════════════════════════════════
function saveScanToJournal(d) {
    if (!state.user) return; // Must be authenticated
    
    try {
        const storageKey = `agrishield_history_${state.user.username}`;
        const history = JSON.parse(localStorage.getItem(storageKey) || '[]');
        
        const scanItem = {
            id: 'scan_' + Date.now() + '_' + Math.floor(Math.random() * 1000),
            timestamp: new Date().toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            }),
            plant_type: d.plant_type || 'Unknown Plant',
            health_status: d.health_status || 'Unknown',
            diagnosis: d.diagnosis || 'Undiagnosed',
            confidence: d.confidence || '0%'
        };
        
        history.unshift(scanItem);
        
        // Cap history size at 15 items
        if (history.length > 15) {
            history.pop();
        }
        
        localStorage.setItem(storageKey, JSON.stringify(history));
    } catch (e) {
        console.warn('Scan history save failed:', e);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SMART TOOLS: CONTROLLER & LOGIC
// ═══════════════════════════════════════════════════════════════════════════
function renderDashboardData() {
    renderScanJournal();
    updateSprayForecast();
    calculateDosage();
    updateCropRotation();
}

// Render Local Journal and Analytics
function renderScanJournal() {
    if (!state.user) return;
    
    const storageKey = `agrishield_history_${state.user.username}`;
    const history = JSON.parse(localStorage.getItem(storageKey) || '[]');
    
    const totalScansEl = document.getElementById('analytics-total-scans');
    const healthRateEl = document.getElementById('analytics-health-rate');
    const prevalentDiseaseEl = document.getElementById('analytics-prevalent-disease');
    const journalList = document.getElementById('journal-list');
    
    // Default analytics
    let totalScans = history.length;
    let healthyCount = 0;
    let diseasedCount = 0;
    let pestCount = 0;
    let nutrientCount = 0;
    let unknownCount = 0;
    let diseaseCounts = {};
    
    history.forEach(item => {
        if (item.health_status === 'Healthy') {
            healthyCount++;
        } else if (item.health_status === 'Diseased') {
            diseasedCount++;
        } else if (item.health_status === 'Pest Infestation') {
            pestCount++;
        } else if (item.health_status === 'Nutrient Deficiency') {
            nutrientCount++;
        } else {
            unknownCount++;
        }
        
        if (item.health_status !== 'Healthy') {
            diseaseCounts[item.diagnosis] = (diseaseCounts[item.diagnosis] || 0) + 1;
        }
    });
    
    let healthRate = totalScans > 0 ? Math.round((healthyCount / totalScans) * 100) : 100;
    
    let prevalentDisease = 'None';
    let maxCount = 0;
    for (const [disease, count] of Object.entries(diseaseCounts)) {
        if (count > maxCount) {
            maxCount = count;
            prevalentDisease = disease;
        }
    }
    
    // Update analytics UI
    if (totalScansEl) totalScansEl.textContent = totalScans;
    if (healthRateEl) {
        healthRateEl.textContent = healthRate + '%';
        const rateBar = document.getElementById('health-rate-bar');
        if (rateBar) rateBar.style.width = healthRate + '%';
    }
    if (prevalentDiseaseEl) prevalentDiseaseEl.textContent = prevalentDisease.slice(0, 24) + (prevalentDisease.length > 24 ? '...' : '');

    // Update dynamic SVG Donut chart partitions
    renderScanHistoryChart(healthyCount, diseasedCount, pestCount, nutrientCount, unknownCount);

    // Populate List
    if (!journalList) return;
    
    if (history.length === 0) {
        journalList.innerHTML = `
            <div class="text-center py-10 border border-dashed border-slate-200 rounded-xl bg-slate-50/50">
                <p class="text-xs text-slate-400 font-semibold">Scan history is empty.</p>
                <p class="text-[10px] text-slate-300 mt-0.5">Diagnosed crops will automatically appear in your journal log.</p>
            </div>
        `;
        return;
    }
    
    const statusIcons = {
        'Healthy': '✅',
        'Diseased': '🔴',
        'Pest Infestation': '🐛',
        'Nutrient Deficiency': '⚠️',
        'Unknown': '❓'
    };
    
    const statusColors = {
        'Healthy': 'text-emerald-700 bg-emerald-50 border-emerald-100',
        'Diseased': 'text-red-700 bg-red-50 border-red-100',
        'Pest Infestation': 'text-orange-700 bg-orange-50 border-orange-100',
        'Nutrient Deficiency': 'text-amber-700 bg-amber-50 border-amber-100',
        'Unknown': 'text-slate-600 bg-slate-50 border-slate-100'
    };

    journalList.innerHTML = history.map(item => {
        const colorClass = statusColors[item.health_status] || statusColors['Unknown'];
        const icon = statusIcons[item.health_status] || statusIcons['Unknown'];
        
        return `
            <div class="bg-white/70 hover:bg-white border border-slate-200/50 rounded-xl p-3.5 transition-all duration-200 hover:shadow-md flex items-center justify-between gap-3 animate-slide-up">
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                        <span class="text-xs font-bold text-slate-800 truncate">${escapeHtml(item.plant_type)}</span>
                        <span class="text-[9px] px-2 py-0.5 rounded-full border font-bold ${colorClass}">
                            ${icon} ${item.health_status}
                        </span>
                    </div>
                    <p class="text-[11px] font-semibold text-slate-500 truncate">${escapeHtml(item.diagnosis)}</p>
                    <p class="text-[9px] text-slate-400 font-medium mt-1">${item.timestamp} · Confidence: ${item.confidence}</p>
                </div>
                <button type="button" onclick="deleteJournalItem('${item.id}')" class="p-1.5 hover:bg-red-50 text-slate-400 hover:text-red-500 rounded-lg transition-colors" title="Delete scan">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                </button>
            </div>
        `;
    }).join('');
}

function deleteJournalItem(id) {
    if (!state.user) return;
    
    try {
        const storageKey = `agrishield_history_${state.user.username}`;
        let history = JSON.parse(localStorage.getItem(storageKey) || '[]');
        history = history.filter(item => item.id !== id);
        localStorage.setItem(storageKey, JSON.stringify(history));
        renderScanJournal();
    } catch (e) {
        console.warn('Failed to delete history item:', e);
    }
}

function clearAllScans() {
    if (!state.user) return;
    if (!confirm('Are you sure you want to completely clear your farm scan journal?')) return;
    
    try {
        const storageKey = `agrishield_history_${state.user.username}`;
        localStorage.setItem(storageKey, '[]');
        renderScanJournal();
    } catch (e) {
        console.warn('Failed to reset history journal:', e);
    }
}

// 🌤️ Simulated Weather Forecast & Spray Advisory Widget
function updateSprayForecast() {
    const climateZone = document.getElementById('climate-zone').value;
    
    // Climate profile rules
    const weatherProfiles = {
        'temperate':   { temp: 21, humidity: 62, wind: 8, soil: 42, condition: 'Clear Skies', risk: 'Low Fungal Danger' },
        'tropical':    { temp: 29, humidity: 88, wind: 12, soil: 78, condition: 'Heavy Dew & Humid', risk: 'High Fungal / Rot Risk' },
        'arid':        { temp: 34, humidity: 21, wind: 18, soil: 14, condition: 'Hot & Dry Winds', risk: 'Evaporative Chemical Loss' },
        'continental': { temp: 16, humidity: 55, wind: 24, soil: 38, condition: 'Strong Winds', risk: 'Drift Danger' }
    };
    
    const profile = weatherProfiles[climateZone] || weatherProfiles['temperate'];
    
    // Update labels in UI
    document.getElementById('weather-temp').textContent = profile.temp + '°C';
    document.getElementById('weather-humidity').textContent = profile.humidity + '%';
    document.getElementById('weather-wind').textContent = profile.wind + ' km/h';
    document.getElementById('weather-soil').textContent = profile.soil + '%';
    document.getElementById('weather-condition').textContent = profile.condition;
    document.getElementById('weather-risk').textContent = profile.risk;
    
    const advisoryEl = document.getElementById('weather-spray-advisory');
    
    // Chemical Spray Safety Rules
    let safetyLabel = 'Optimal Spray Window';
    let safetyDesc = 'Temperature, humidity, and wind speeds are perfect. Sprayed chemicals will dry safely with zero drift or excessive evaporation.';
    let badgeColor = 'text-emerald-700 bg-emerald-50 border-emerald-200';
    
    if (profile.wind > 20) {
        safetyLabel = 'Danger: High Wind Drift';
        safetyDesc = 'Warning: Wind speeds exceed 20 km/h. Avoid chemical spraying now. Fungicides and pesticides will drift off-target, contaminating adjacent fields and reducing effectiveness.';
        badgeColor = 'text-red-700 bg-red-50 border-red-200';
    } else if (profile.temp > 32) {
        safetyLabel = 'Caution: Heat Evaporation';
        safetyDesc = 'High heat simulated. Liquid chemicals will evaporate rapidly before absorption, leading to chemical leaf scorch and reduced efficiency. Spraying during cooler early mornings is advised.';
        badgeColor = 'text-amber-700 bg-amber-50 border-amber-200';
    } else if (profile.humidity > 85) {
        safetyLabel = 'Caution: High Fungal Moisture';
        safetyDesc = 'Excessive humidity will slow chemical drying times, increasing wash-off risks if a summer storm hits. Leaves are damp: monitor crop surfaces carefully.';
        badgeColor = 'text-amber-700 bg-amber-50 border-amber-200';
    }
    
    if (advisoryEl) {
        advisoryEl.innerHTML = `
            <div class="flex items-center justify-between mb-2">
                <span class="text-xs font-bold text-slate-800">Spraying Window Advisory</span>
                <span id="weather-spray-badge" class="text-[10px] px-2.5 py-0.5 rounded-full border font-bold ${badgeColor}">
                    ${safetyLabel}
                </span>
            </div>
            <p class="text-[11px] text-slate-500 leading-relaxed font-semibold">${safetyDesc}</p>
        `;
    }
}

// 🧪 Dosage & Treatment Calculator
function calculateDosage() {
    const calcCrop = document.getElementById('calc-crop').value;
    const calcSize = parseFloat(document.getElementById('calc-size').value) || 1;
    const sizeUnit = document.getElementById('calc-unit').value;
    
    // Normalize field size to Acres for the baseline math
    const sizeInAcres = sizeUnit === 'hectares' ? calcSize * 2.471 : calcSize;
    
    // Agronomic databases for standard treatments
    const treatments = {
        'tomato_blight': {
            name: 'Chlorothalonil 720SC Fungicide',
            dosePerL: 2.5, // ml per liter
            waterPerAcre: 200, // liters per acre
            costPerAcre: 15.00, // USD per acre
            phi: 7, // days
            protocol: 'Wear standard N95 mask, rubber gloves, and protective suit. Avoid spraying within 50 feet of open waterways.',
            nozzle: 'Flat Fan (XR11004)',
            pressure: '2.8 bar',
            dilution: '1:400 (Liquid)'
        },
        'potato_blight': {
            name: 'Mancozeb 75DF Wettable Powder',
            dosePerL: 3.0, // grams per liter
            waterPerAcre: 250, // liters per acre
            costPerAcre: 18.50, // USD
            phi: 14,
            protocol: 'Highly toxic if inhaled. Apply only in light wind. Wash hands thoroughly before drinking or eating.',
            nozzle: 'Hollow Cone (TX-VK8)',
            pressure: '3.5 bar',
            dilution: '1:330 (Powder)'
        },
        'corn_rust': {
            name: 'Tebuconazole 250EC Fungicide',
            dosePerL: 2.0, // ml per liter
            waterPerAcre: 150, // liters per acre
            costPerAcre: 22.00,
            phi: 21,
            protocol: 'Systemic triazole chemical. Allow 21 days safety interval before grain harvest or silage chopping.',
            nozzle: 'Dual Flat Fan (AITTJ60)',
            pressure: '3.0 bar',
            dilution: '1:500 (Liquid)'
        },
        'apple_scab': {
            name: 'Captan 50WP Broad-Spectrum Powder',
            dosePerL: 1.5, // grams per liter
            waterPerAcre: 400, // liters per acre
            costPerAcre: 35.00,
            phi: 14,
            protocol: 'Fruit-zone canopy application. Avoid application during direct high-intensity sun to prevent fruit russeting.',
            nozzle: 'Air Induction Fan (AIXR110)',
            pressure: '4.0 bar',
            dilution: '1:660 (Powder)'
        },
        'general_pest': {
            name: 'Cold-Pressed Concentrated Neem Oil (Organic)',
            dosePerL: 5.0, // ml per liter
            waterPerAcre: 200, // liters per acre
            costPerAcre: 12.00,
            phi: 0,
            protocol: '100% Organic solution. Dilute with warm water and 1 tsp of mild dish soap to emulsify. Safe for bees if applied in evening.',
            nozzle: 'Flat Fan (XR11003)',
            pressure: '2.0 bar',
            dilution: '1:200 (Organic)'
        }
    };
    
    const cropRx = treatments[calcCrop] || treatments['general_pest'];
    
    // Perform calculations
    const totalWater = Math.round(cropRx.waterPerAcre * sizeInAcres);
    const totalIngredient = (cropRx.dosePerL * totalWater).toFixed(0);
    const totalCost = (cropRx.costPerAcre * sizeInAcres).toFixed(2);
    
    const isLiquid = calcCrop !== 'potato_blight' && calcCrop !== 'apple_scab';
    const ingredientUnit = isLiquid ? 'mL' : 'g';
    const activeIngredientKg = isLiquid ? (totalIngredient / 1000).toFixed(2) + ' Liters' : (totalIngredient / 1000).toFixed(2) + ' kg';
    
    // Update Calculator UI
    document.getElementById('calc-water-val').textContent = totalWater + ' Liters';
    document.getElementById('calc-ingredient-val').textContent = activeIngredientKg;
    document.getElementById('calc-cost-val').textContent = '$' + totalCost;
    
    const phiEl = document.getElementById('calc-phi-val');
    if (phiEl) {
        if (cropRx.phi > 0) {
            phiEl.className = 'text-sm font-black text-red-600 bg-red-50 border border-red-200/50 px-3 py-1 rounded-full animate-pulse';
            phiEl.innerHTML = `⚠️ ${cropRx.phi} Days PHI`;
        } else {
            phiEl.className = 'text-sm font-black text-emerald-700 bg-emerald-50 border border-emerald-200/50 px-3 py-1 rounded-full';
            phiEl.innerHTML = `✅ 0 Days (Organic)`;
        }
    }
    
    document.getElementById('calc-name-val').textContent = cropRx.name;
    document.getElementById('calc-protocol').textContent = cropRx.protocol;
    
    // Inject advanced delivery recommendations
    const nozzleVal = document.getElementById('calc-nozzle-val');
    const pressureVal = document.getElementById('calc-pressure-val');
    const dilutionVal = document.getElementById('calc-dilution-val');
    if (nozzleVal) nozzleVal.textContent = cropRx.nozzle;
    if (pressureVal) pressureVal.textContent = cropRx.pressure;
    if (dilutionVal) dilutionVal.textContent = cropRx.dilution;
}

// 🔄 Interactive Crop Rotation Planner
function updateCropRotation() {
    const currentCrop = document.getElementById('rotation-start').value;
    
    const rotationDatabases = {
        'tomato': [
            { year: 1, name: '🍅 Tomatoes / Peppers (Active)', category: 'Heavy Feeders (Solanaceae)', nitrogen: -50, note: 'Depletes soil calcium and nitrogen. Leaves fungal spores (early blight) in crop residues.' },
            { year: 2, name: '🫘 Soybeans / Bush Beans', category: 'Nitrogen-Fixing Cover Crop (Legumes)', nitrogen: 80, note: 'Symbiotic rhizobia bacteria capture atmospheric nitrogen, enriching depleted soil naturally. Disrupts tomato pathogens.' },
            { year: 3, name: '🥬 Cabbage / Broccoli / Kale', category: 'Moderate Feeders (Brassicas)', nitrogen: -25, note: 'Utilizes remaining nitrogen. Taproot feeds at a moderate depth, preventing erosion.' },
            { year: 4, name: '🥕 Carrots / Radishes / Garlic', category: 'Light Feeders (Root Crops)', nitrogen: -10, note: 'Break up clay soil compacted by previous rotations. Tap deep potassium reserves. Prepares soil structure for tomatoes.' }
        ],
        'corn': [
            { year: 1, name: '🌽 Corn / Maize (Active)', category: 'High Nitrogen Consumers (Poaceae)', nitrogen: -120, note: 'Extremely heavy grass crop. Requires massive nitrogen reserves and leaves high carbohydrate stalks.' },
            { year: 2, name: '🫘 Green Peas / Alfalfa / Clover', category: 'Soil Nitrogen Replenishers (Legumes)', nitrogen: 150, note: 'Essential to capture nitrogen, restoring organic carbon blocks depleted by heavy corn stalks.' },
            { year: 3, name: '🎃 Pumpkin / Squash / Melons', category: 'Ground Covers (Cucurbits)', nitrogen: -30, note: 'Large sprawling vine leaves act as a natural organic green-mulch, suppressing weed growth and moisture evaporation.' },
            { year: 4, name: '🥔 Potatoes / Sweet Potatoes', category: 'Root Tubers (Solanaceae)', nitrogen: -40, note: 'Aerates sub-soil during tuber expansion. Utilizes deep soil nutrients untouched by corn roots.' }
        ],
        'potato': [
            { year: 1, name: '🥔 Potatoes (Active)', category: 'Root Tubers (Solanaceae)', nitrogen: -60, note: 'Consumes potassium. Increases risks of subterranean common scab and early blight spores.' },
            { year: 2, name: '🫘 Crimson Clover Cover', category: 'Nitrogen-Fixer / Soil Builder', nitrogen: 100, note: 'Restores nitrogen. Provides thick root mass that breaks down to replenish organic matter before next rotation.' },
            { year: 3, name: '🥬 Broccoli / Mustard Greens', category: 'Glucosinolate Producers (Brassicas)', nitrogen: -15, note: 'Acts as a natural bio-fumigant when plowed back into the soil, killing scab bacteria and nematodes.' },
            { year: 4, name: '🧅 Onions / Garlic / Leeks', category: 'Shallow Bulb Crops (Alliaceae)', nitrogen: -10, note: 'Shallow root system lets deeper layers recover. Natural sulfur root secretions repel subterranean insects.' }
        ],
        'legume': [
            { year: 1, name: '🫘 Legumes / Beans (Active)', category: 'Nitrogen Builders (Fabaceae)', nitrogen: 90, note: 'Replenishes soil, adding highly soluble nitrates through root nodule degradation.' },
            { year: 2, name: '🥬 Kale / Brussels Sprouts', category: 'Heavy Nitrogen Consumers (Brassicas)', nitrogen: -45, note: 'Directly benefits from the nitrated soil left by legumes, growing massive healthy foliage.' },
            { year: 3, name: '🥕 Carrots / Parsnips', category: 'Root Minerals Collectors', nitrogen: -15, note: 'Collects remaining minerals. Slows fungal spore accumulation due to unrelated crop family.' },
            { year: 4, name: '🍅 Tomatoes / Eggplants', category: 'Heavy Feeders (Solanaceae)', nitrogen: -55, note: 'Enjoys highly structured, nutrient-rich soil prepared by the previous rotations.' }
        ]
    };
    
    const rotation = rotationDatabases[currentCrop] || rotationDatabases['tomato'];
    const timelineEl = document.getElementById('rotation-timeline');
    
    if (!timelineEl) return;
    
    let alertHtml = '';
    if (currentCrop === 'tomato') {
        alertHtml = `
            <div class="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-2.5 animate-slide-up">
                <span class="text-base mt-0.5">⚠️</span>
                <div>
                    <h5 class="text-xs font-bold text-amber-800">Solanaceae Pathogen Warning</h5>
                    <p class="text-[10px] text-amber-600 mt-0.5 leading-relaxed font-semibold">Avoid planting **Potatoes** or **Peppers** in Year 2. They share the same fungal pathogens (Bacterial Spot, Alternaria Blight). Spores will infect the crop immediately, wiping out the benefit of the crop cycle.</p>
                </div>
            </div>
        `;
    } else if (currentCrop === 'potato') {
        alertHtml = `
            <div class="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-2.5 animate-slide-up">
                <span class="text-base mt-0.5">⚠️</span>
                <div>
                    <h5 class="text-xs font-bold text-amber-800">Tuber Scab Spore Accumulation Alert</h5>
                    <p class="text-[10px] text-amber-600 mt-0.5 leading-relaxed font-semibold">Avoid planting **Tomatoes** or **Eggplants** directly after Potatoes. Fungal scab spores thrive in soil for years. Follow the recommended bio-fumigant Brassica schedule (Year 3) to disinfect the field first.</p>
                </div>
            </div>
        `;
    }

    timelineEl.innerHTML = `
        <div class="relative pl-6 border-l-2 border-slate-100 space-y-5">
            ${rotation.map(item => `
                <div class="relative animate-slide-up">
                    <!-- Dot Badge -->
                    <span class="absolute -left-[31px] top-0.5 w-4 h-4 rounded-full border-2 border-white flex items-center justify-center text-[9px] font-black font-heading text-white ${item.year === 1 ? 'bg-emerald-600' : 'bg-slate-300'} shadow-sm">
                        ${item.year}
                    </span>
                    <div>
                        <div class="flex items-center gap-2">
                            <h5 class="text-xs font-bold text-slate-800">${item.name}</h5>
                            <span class="text-[9px] px-2 py-0.5 bg-slate-50 border border-slate-200/50 rounded-full font-bold text-slate-400 uppercase tracking-wide">Year ${item.year}</span>
                            <span class="text-[9px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wide border ${item.nitrogen >= 0 ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-rose-50 text-rose-700 border-rose-200'}">
                                ${item.nitrogen >= 0 ? '+' : ''}${item.nitrogen} lbs N/acre
                            </span>
                        </div>
                        <p class="text-[10px] font-bold text-emerald-700 mt-0.5">${item.category}</p>
                        <p class="text-[11px] text-slate-500 leading-relaxed mt-1 font-semibold">${item.note}</p>
                    </div>
                </div>
            `).join('')}
        </div>
        ${alertHtml}
    `;
}

function togglePasswordVisibility(inputId, svgId) {
    const input = document.getElementById(inputId);
    const svg = document.getElementById(svgId);
    if (!input || !svg) return;
    
    if (input.type === 'password') {
        input.type = 'text';
        // Eye-off icon
        svg.innerHTML = `
            <path stroke-linecap="round" stroke-linejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
        `;
    } else {
        input.type = 'password';
        // Normal Eye icon
        svg.innerHTML = `
            <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
        `;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2FA CLIENT-SIDE SECURITY CONTROLLERS & INTERVALS
// ═══════════════════════════════════════════════════════════════════════════
let twoFactorTimerInterval = null;

function setupOTPInputNavigation() {
    const inputs = document.querySelectorAll('#otp-inputs-container .otp-digit');
    if (!inputs.length) return;
    
    inputs.forEach((input, index) => {
        input.addEventListener('focus', () => {
            input.select();
        });

        input.addEventListener('input', (e) => {
            const val = e.target.value;
            e.target.value = val.replace(/[^0-9]/g, ''); // Numeric only
            
            if (e.target.value.length === 1 && index < inputs.length - 1) {
                inputs[index + 1].focus();
            }
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Backspace' && !e.target.value && index > 0) {
                inputs[index - 1].focus();
                inputs[index - 1].value = '';
            } else if (e.key === 'ArrowLeft' && index > 0) {
                inputs[index - 1].focus();
            } else if (e.key === 'ArrowRight' && index < inputs.length - 1) {
                inputs[index + 1].focus();
            }
        });
        
        input.addEventListener('paste', (e) => {
            e.preventDefault();
            const pasteData = (e.clipboardData || window.clipboardData).getData('text').trim();
            if (/^\d{6}$/.test(pasteData)) {
                inputs.forEach((inp, idx) => {
                    inp.value = pasteData[idx];
                });
                inputs[5].focus();
            }
        });
    });
}

function start2FATimer(durationSeconds) {
    if (twoFactorTimerInterval) {
        clearInterval(twoFactorTimerInterval);
    }
    
    const timerDisplay = document.getElementById('auth-2fa-timer');
    const resendBtn = document.getElementById('auth-2fa-resend-btn');
    if (!timerDisplay || !resendBtn) return;
    
    resendBtn.disabled = true;
    resendBtn.classList.add('opacity-40', 'cursor-not-allowed');
    
    let timeLeft = durationSeconds;
    
    function updateTimer() {
        const mins = Math.floor(timeLeft / 60);
        const secs = timeLeft % 60;
        timerDisplay.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        
        if (timeLeft <= 0) {
            clearInterval(twoFactorTimerInterval);
            timerDisplay.textContent = 'Expired';
            resendBtn.disabled = false;
            resendBtn.classList.remove('opacity-40', 'cursor-not-allowed');
        }
        timeLeft--;
    }
    
    updateTimer();
    twoFactorTimerInterval = setInterval(updateTimer, 1000);
}

function cancel2FA() {
    if (twoFactorTimerInterval) {
        clearInterval(twoFactorTimerInterval);
    }
    
    const credentialsCont = dom.authCredentialsContainer();
    const container2FA = dom.auth2FAContainer();
    
    if (credentialsCont) credentialsCont.classList.remove('hidden');
    if (container2FA) container2FA.classList.add('hidden');
    
    // Reset fields
    const inputs = document.querySelectorAll('#otp-inputs-container .otp-digit');
    inputs.forEach(inp => inp.value = '');
    
    const errorBox = document.getElementById('auth-2fa-error');
    if (errorBox) {
        errorBox.classList.add('hidden');
        errorBox.textContent = '';
    }
    
    dom.authError().classList.add('hidden');
    dom.authSpinner().classList.add('hidden');
    dom.authBtnText().textContent = state.authMode === 'login' ? 'Sign In' : 'Register';
}

async function resend2FACode() {
    const errorBox = document.getElementById('auth-2fa-error');
    const resendBtn = document.getElementById('auth-2fa-resend-btn');
    if (!errorBox || !resendBtn) return;
    
    errorBox.classList.add('hidden');
    errorBox.textContent = '';
    resendBtn.disabled = true;
    resendBtn.classList.add('opacity-40');
    
    try {
        const response = await fetch('/api/auth/resend_2fa', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            start2FATimer(600); // Restart 10 mins
            
            const inputs = document.querySelectorAll('#otp-inputs-container .otp-digit');
            inputs.forEach(inp => inp.value = '');
            if (inputs[0]) inputs[0].focus();
            
            errorBox.classList.remove('hidden');
            errorBox.classList.replace('text-red-600', 'text-emerald-700');
            errorBox.classList.replace('bg-red-50', 'bg-emerald-50');
            errorBox.classList.replace('border-red-100', 'border-emerald-100');
            errorBox.textContent = 'A new 6-digit verification code has been dispatched!';
            
            setTimeout(() => {
                errorBox.classList.add('hidden');
                errorBox.classList.replace('text-emerald-700', 'text-red-600');
                errorBox.classList.replace('bg-emerald-50', 'bg-red-50');
                errorBox.classList.replace('border-emerald-100', 'border-red-100');
            }, 4000);
        } else {
            errorBox.textContent = data.error || 'Failed to resend code.';
            errorBox.classList.remove('hidden');
            resendBtn.disabled = false;
            resendBtn.classList.remove('opacity-40');
        }
    } catch (e) {
        errorBox.textContent = 'Service unavailable. Please try again.';
        errorBox.classList.remove('hidden');
        resendBtn.disabled = false;
        resendBtn.classList.remove('opacity-40');
    }
}

async function handle2FAVerification(e) {
    e.preventDefault();
    
    const errorBox = document.getElementById('auth-2fa-error');
    const spinner = document.getElementById('auth-2fa-spinner');
    const verifyBtn = document.getElementById('auth-2fa-verify-btn');
    if (!errorBox || !spinner || !verifyBtn) return;
    
    errorBox.classList.add('hidden');
    errorBox.textContent = '';
    
    const inputs = document.querySelectorAll('#otp-inputs-container .otp-digit');
    let codeStr = '';
    inputs.forEach(inp => {
        codeStr += inp.value.trim();
    });
    
    if (codeStr.length !== 6) {
        errorBox.textContent = 'Please enter all 6 digits of the code.';
        errorBox.classList.remove('hidden');
        return;
    }
    
    spinner.classList.remove('hidden');
    verifyBtn.disabled = true;
    
    try {
        const response = await fetch('/api/auth/verify_2fa', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ otp: codeStr })
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (twoFactorTimerInterval) {
                clearInterval(twoFactorTimerInterval);
            }
            
            state.user = data.user;
            setupAuthenticatedUI();
            
            // Welcome msg in chatbot
            if (state.authMode === 'signup') {
                appendChatBubble('ai', `👋 **Welcome, ${state.user.username}!** Thank you for registering. I'm your AgriShield AI advisor. I can help you with leaf/fruit diagnoses, dosage estimations, soil moisture issues, and planting rotations. Try out our new **Smart Tools** tab for active field calculators!`);
            }
            
            // Clean up
            dom.authUsername().value = '';
            dom.authEmail().value = '';
            dom.authPassword().value = '';
            inputs.forEach(inp => inp.value = '');
            
            // Revert containers back to credentials panel
            const credentialsCont = dom.authCredentialsContainer();
            const container2FA = dom.auth2FAContainer();
            if (credentialsCont) credentialsCont.classList.remove('hidden');
            if (container2FA) container2FA.classList.add('hidden');
            
            dom.authOverlay().classList.add('hidden');
        } else {
            errorBox.textContent = data.error || 'Verification failed. Incorrect code.';
            errorBox.classList.remove('hidden');
        }
    } catch (err) {
        errorBox.textContent = 'Communication failure. Please verify connection.';
        errorBox.classList.remove('hidden');
    } finally {
        spinner.classList.add('hidden');
        verifyBtn.disabled = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GEOMETRIC CLIENT-SIDE SVG DONUT CHART ENGINE
// ═══════════════════════════════════════════════════════════════════════════
function renderScanHistoryChart(healthy, diseased, pests, nutrients, unknowns) {
    const slicesGroup = document.getElementById('donut-slices-group');
    const centerRate = document.getElementById('donut-center-rate');
    const legendHealthy = document.getElementById('legend-healthy-count');
    const legendDiseased = document.getElementById('legend-diseased-count');
    const legendPests = document.getElementById('legend-pests-count');
    const legendNutrient = document.getElementById('legend-nutrient-count');
    
    if (legendHealthy) legendHealthy.textContent = healthy;
    if (legendDiseased) legendDiseased.textContent = diseased;
    if (legendPests) legendPests.textContent = pests;
    if (legendNutrient) legendNutrient.textContent = nutrients;
    
    const total = healthy + diseased + pests + nutrients + unknowns;
    const healthRate = total > 0 ? Math.round((healthy / total) * 100) : 100;
    if (centerRate) {
        centerRate.textContent = total > 0 ? healthRate + '%' : '100%';
    }
    
    if (!slicesGroup) return;
    slicesGroup.innerHTML = '';
    
    const circumference = 238.76;
    
    if (total === 0) {
        // Base placeholder slice (emerald base)
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', '48');
        circle.setAttribute('cy', '48');
        circle.setAttribute('r', '38');
        circle.setAttribute('fill', 'transparent');
        circle.setAttribute('stroke', '#10b981'); // emerald-500
        circle.setAttribute('stroke-width', '10');
        circle.setAttribute('stroke-dasharray', `${circumference} 0`);
        circle.setAttribute('stroke-dashoffset', '0');
        slicesGroup.appendChild(circle);
        return;
    }
    
    const categories = [
        { count: healthy, color: '#10b981' },    // Healthy (emerald)
        { count: diseased, color: '#ef4444' },   // Diseased (red)
        { count: pests, color: '#f97316' },      // Pests (orange)
        { count: nutrients, color: '#f59e0b' },  // Deficiencies (amber)
        { count: unknowns, color: '#64748b' }    // Unknown (slate)
    ];
    
    let currentOffset = 0;
    
    categories.forEach(cat => {
        if (cat.count === 0) return;
        
        const percentage = cat.count / total;
        const sliceLength = percentage * circumference;
        const remainingLength = circumference - sliceLength;
        
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', '48');
        circle.setAttribute('cy', '48');
        circle.setAttribute('r', '38');
        circle.setAttribute('fill', 'transparent');
        circle.setAttribute('stroke', cat.color);
        circle.setAttribute('stroke-width', '10');
        circle.setAttribute('stroke-dasharray', `${sliceLength} ${remainingLength}`);
        circle.setAttribute('stroke-dashoffset', -currentOffset);
        circle.style.transition = 'stroke-dashoffset 0.5s ease-in-out';
        
        slicesGroup.appendChild(circle);
        
        currentOffset += sliceLength;
    });
}
