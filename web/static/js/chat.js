// ===== CHAT JAVASCRIPT =====

// Chat state
let chatState = {
    messages: [],
    isTyping: false,
    currentModel: 'gpt-4',
    temperature: 0.7,
    maxTokens: 2048,
    memoryContext: 5,
    attachedFiles: [],
    isRecording: false,
    mediaRecorder: null,
    audioChunks: []
};

// Initialize chat
document.addEventListener('DOMContentLoaded', function() {
    initializeChat();
});

function initializeChat() {
    initializeInput();
    initializeSettings();
    initializeFileUpload();
    initializeVoiceInput();
    initializeQuickActions();
    loadChatHistory();
}

// Input functionality
function initializeInput() {
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const charCount = document.getElementById('char-count');

    if (messageInput) {
        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            
            // Update character count
            if (charCount) {
                charCount.textContent = this.value.length;
            }
            
            // Enable/disable send button
            if (sendButton) {
                sendButton.disabled = this.value.trim().length === 0;
            }
        });

        // Send message on Enter (Shift+Enter for new line)
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Focus input on page load
        messageInput.focus();
    }
}

function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (!message || chatState.isTyping) return;

    // Add user message
    addMessage(message, 'user');
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Update character count
    const charCount = document.getElementById('char-count');
    if (charCount) {
        charCount.textContent = '0';
    }

    // Disable send button
    const sendButton = document.getElementById('send-button');
    if (sendButton) {
        sendButton.disabled = true;
    }

    // Show typing indicator
    showTypingIndicator();

    // Send to AI
    sendToAI(message);
}

function addMessage(content, sender, timestamp = new Date()) {
    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const timeString = formatTime(timestamp);
    const avatarIcon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
    const authorName = sender === 'user' ? 'You' : 'AI Assistant';

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="${avatarIcon}"></i>
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-author">${authorName}</span>
                <span class="message-time">${timeString}</span>
            </div>
            <div class="message-text">${formatMessageContent(content)}</div>
            <div class="message-actions">
                <button class="message-action" onclick="copyMessage(this)" title="Copy">
                    <i class="fas fa-copy"></i>
                </button>
                ${sender === 'ai' ? '<button class="message-action" onclick="regenerateResponse()" title="Regenerate"><i class="fas fa-redo"></i></button>' : ''}
            </div>
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    
    // Add to chat state
    chatState.messages.push({
        content,
        sender,
        timestamp
    });

    // Limit memory context
    if (chatState.messages.length > chatState.memoryContext * 2) {
        chatState.messages = chatState.messages.slice(-chatState.memoryContext * 2);
    }
}

function formatMessageContent(content) {
    // Convert URLs to links
    content = content.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Convert code blocks
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="$1">$2</code></pre>');
    
    // Convert inline code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Convert line breaks
    content = content.replace(/\n/g, '<br>');
    
    return content;
}

function formatTime(date) {
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) { // Less than 1 minute
        return 'Just now';
    } else if (diff < 3600000) { // Less than 1 hour
        const minutes = Math.floor(diff / 60000);
        return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    } else if (diff < 86400000) { // Less than 1 day
        const hours = Math.floor(diff / 3600000);
        return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else {
        return date.toLocaleDateString();
    }
}

function scrollToBottom() {
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Typing indicator
function showTypingIndicator() {
    chatState.isTyping = true;
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.classList.add('active');
        scrollToBottom();
    }
}

function hideTypingIndicator() {
    chatState.isTyping = false;
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.classList.remove('active');
    }
}

// AI Communication
async function sendToAI(message) {
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                model: chatState.currentModel,
                temperature: chatState.temperature,
                max_tokens: chatState.maxTokens,
                context: chatState.messages.slice(-chatState.memoryContext * 2)
            })
        });

        if (!response.ok) {
            throw new Error('Failed to get response from AI');
        }

        const data = await response.json();
        hideTypingIndicator();
        
        if (data.response) {
            addMessage(data.response, 'ai');
        } else {
            addMessage('Sorry, I encountered an error. Please try again.', 'ai');
        }

    } catch (error) {
        console.error('Error sending message to AI:', error);
        hideTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.', 'ai');
        showNotification('Failed to connect to AI service', 'error');
    }
}

// Settings functionality
function initializeSettings() {
    const modelSelect = document.getElementById('model-select');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const maxTokensInput = document.getElementById('max-tokens');
    const memorySlider = document.getElementById('memory-context');
    const memoryValue = document.getElementById('memory-value');

    if (modelSelect) {
        modelSelect.value = chatState.currentModel;
        modelSelect.addEventListener('change', function() {
            chatState.currentModel = this.value;
            showNotification(`Model changed to ${this.value}`, 'info');
        });
    }

    if (temperatureSlider && temperatureValue) {
        temperatureSlider.value = chatState.temperature;
        temperatureValue.textContent = chatState.temperature;
        temperatureSlider.addEventListener('input', function() {
            chatState.temperature = parseFloat(this.value);
            temperatureValue.textContent = this.value;
        });
    }

    if (maxTokensInput) {
        maxTokensInput.value = chatState.maxTokens;
        maxTokensInput.addEventListener('change', function() {
            chatState.maxTokens = parseInt(this.value);
        });
    }

    if (memorySlider && memoryValue) {
        memorySlider.value = chatState.memoryContext;
        memoryValue.textContent = `${chatState.memoryContext} messages`;
        memorySlider.addEventListener('input', function() {
            chatState.memoryContext = parseInt(this.value);
            memoryValue.textContent = `${this.value} messages`;
        });
    }
}

function toggleSettings() {
    const settings = document.getElementById('chat-settings');
    if (settings) {
        settings.classList.toggle('active');
    }
}

// File upload functionality
function initializeFileUpload() {
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('file-upload-area');

    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('drop', handleFileDrop);
    }

    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.style.borderColor = 'var(--primary-color)';
    e.currentTarget.style.background = 'rgba(0, 212, 255, 0.05)';
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.style.borderColor = 'var(--border-color)';
    e.currentTarget.style.background = 'transparent';
    
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    handleFiles(files);
}

function handleFiles(files) {
    const fileList = document.getElementById('file-list');
    if (!fileList) return;

    files.forEach(file => {
        if (file.size > 10 * 1024 * 1024) { // 10MB limit
            showNotification(`File ${file.name} is too large. Maximum size is 10MB.`, 'error');
            return;
        }

        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <i class="fas fa-file"></i>
                <span>${file.name}</span>
                <span class="file-size">${formatBytes(file.size)}</span>
            </div>
            <button class="file-remove" onclick="removeFile(this, '${file.name}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        fileList.appendChild(fileItem);

        chatState.attachedFiles.push(file);
    });
}

function removeFile(button, fileName) {
    const fileItem = button.parentElement;
    fileItem.remove();
    
    chatState.attachedFiles = chatState.attachedFiles.filter(file => file.name !== fileName);
}

function attachFile() {
    const fileModal = document.getElementById('file-modal');
    if (fileModal) {
        fileModal.classList.add('active');
    }
}

function closeFileModal() {
    const fileModal = document.getElementById('file-modal');
    if (fileModal) {
        fileModal.classList.remove('active');
    }
}

function uploadFiles() {
    if (chatState.attachedFiles.length === 0) {
        showNotification('No files selected', 'warning');
        return;
    }

    // Here you would typically upload files to your server
    // For now, we'll just show a success message
    showNotification(`${chatState.attachedFiles.length} file(s) attached`, 'success');
    closeFileModal();
}

// Voice input functionality
function initializeVoiceInput() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        chatState.recognition = new SpeechRecognition();
        chatState.recognition.continuous = true;
        chatState.recognition.interimResults = true;
        chatState.recognition.lang = 'en-US';

        chatState.recognition.onresult = function(event) {
            let transcript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    transcript += event.results[i][0].transcript;
                }
            }
            
            if (transcript) {
                const transcriptElement = document.getElementById('voice-transcript');
                if (transcriptElement) {
                    transcriptElement.innerHTML = `<p>${transcript}</p>`;
                }
                
                const sendVoiceBtn = document.getElementById('send-voice-btn');
                if (sendVoiceBtn) {
                    sendVoiceBtn.disabled = false;
                }
            }
        };

        chatState.recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            stopVoiceRecording();
            showNotification('Voice recognition error', 'error');
        };
    }
}

function voiceInput() {
    const voiceModal = document.getElementById('voice-modal');
    if (voiceModal) {
        voiceModal.classList.add('active');
    }
}

function closeVoiceModal() {
    const voiceModal = document.getElementById('voice-modal');
    if (voiceModal) {
        voiceModal.classList.remove('active');
        stopVoiceRecording();
    }
}

function toggleVoiceRecording() {
    if (chatState.isRecording) {
        stopVoiceRecording();
    } else {
        startVoiceRecording();
    }
}

function startVoiceRecording() {
    if (!chatState.recognition) {
        showNotification('Voice recognition not supported', 'error');
        return;
    }

    chatState.isRecording = true;
    const voiceButton = document.getElementById('voice-button');
    const visualizer = document.getElementById('voice-visualizer');
    const transcript = document.getElementById('voice-transcript');

    if (voiceButton) {
        voiceButton.classList.add('recording');
        voiceButton.innerHTML = '<i class="fas fa-stop"></i><span>Stop Recording</span>';
    }

    if (visualizer) {
        visualizer.style.display = 'flex';
    }

    if (transcript) {
        transcript.innerHTML = '<p>Listening...</p>';
    }

    chatState.recognition.start();
}

function stopVoiceRecording() {
    chatState.isRecording = false;
    const voiceButton = document.getElementById('voice-button');
    const visualizer = document.getElementById('voice-visualizer');

    if (voiceButton) {
        voiceButton.classList.remove('recording');
        voiceButton.innerHTML = '<i class="fas fa-microphone"></i><span>Start Recording</span>';
    }

    if (visualizer) {
        visualizer.style.display = 'none';
    }

    if (chatState.recognition) {
        chatState.recognition.stop();
    }
}

function sendVoiceMessage() {
    const transcriptElement = document.getElementById('voice-transcript');
    if (transcriptElement) {
        const transcript = transcriptElement.textContent.trim();
        if (transcript && transcript !== 'Your speech will appear here...' && transcript !== 'Listening...') {
            closeVoiceModal();
            
            const messageInput = document.getElementById('message-input');
            if (messageInput) {
                messageInput.value = transcript;
                messageInput.dispatchEvent(new Event('input'));
            }
        }
    }
}

// Quick actions
function initializeQuickActions() {
    // Quick actions are handled by onclick attributes in HTML
}

function quickPrompt(prompt) {
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.value = prompt;
        messageInput.focus();
        messageInput.dispatchEvent(new Event('input'));
    }
}

// Utility functions
function copyMessage(button) {
    const messageText = button.closest('.message').querySelector('.message-text');
    const text = messageText.textContent || messageText.innerText;
    
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Message copied to clipboard', 'success');
    }).catch(() => {
        showNotification('Failed to copy message', 'error');
    });
}

function regenerateResponse() {
    if (chatState.messages.length > 0) {
        const lastUserMessage = chatState.messages.filter(m => m.sender === 'user').pop();
        if (lastUserMessage) {
            // Remove last AI response
            const messagesContainer = document.getElementById('chat-messages');
            const lastMessage = messagesContainer.lastElementChild;
            if (lastMessage && lastMessage.classList.contains('ai-message')) {
                lastMessage.remove();
                chatState.messages.pop();
            }
            
            // Regenerate response
            showTypingIndicator();
            sendToAI(lastUserMessage.content);
        }
    }
}

function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        const messagesContainer = document.getElementById('chat-messages');
        if (messagesContainer) {
            messagesContainer.innerHTML = `
                <div class="message ai-message">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <div class="message-header">
                            <span class="message-author">AI Assistant</span>
                            <span class="message-time">Just now</span>
                        </div>
                        <div class="message-text">
                            Hello! I'm your AI assistant. I'm here to help you with any questions, tasks, or creative projects. What would you like to work on today?
                        </div>
                        <div class="message-actions">
                            <button class="message-action" onclick="copyMessage(this)">
                                <i class="fas fa-copy"></i>
                            </button>
                            <button class="message-action" onclick="regenerateResponse()">
                                <i class="fas fa-redo"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        }
        
        chatState.messages = [];
        showNotification('Chat history cleared', 'info');
    }
}

function exportChat() {
    const chatData = {
        messages: chatState.messages,
        settings: {
            model: chatState.currentModel,
            temperature: chatState.temperature,
            maxTokens: chatState.maxTokens,
            memoryContext: chatState.memoryContext
        },
        exportDate: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aivance-chat-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('Chat exported successfully', 'success');
}

function clearInput() {
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.value = '';
        messageInput.style.height = 'auto';
        messageInput.dispatchEvent(new Event('input'));
    }
}

// Load chat history from localStorage
function loadChatHistory() {
    try {
        const saved = localStorage.getItem('aivance-chat-history');
        if (saved) {
            const data = JSON.parse(saved);
            if (data.messages && Array.isArray(data.messages)) {
                chatState.messages = data.messages;
                data.messages.forEach(msg => {
                    addMessage(msg.content, msg.sender, new Date(msg.timestamp));
                });
            }
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// Save chat history to localStorage
function saveChatHistory() {
    try {
        localStorage.setItem('aivance-chat-history', JSON.stringify({
            messages: chatState.messages,
            timestamp: new Date().toISOString()
        }));
    } catch (error) {
        console.error('Error saving chat history:', error);
    }
}

// Auto-save chat history
setInterval(saveChatHistory, 30000); // Save every 30 seconds

// Format bytes utility
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
} 