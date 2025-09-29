class ChatCSVApp {
    constructor() {
        this.llmProviders = {};
        this.currentStep = 1;
        this.isLLMConfigured = false;
        this.areFilesUploaded = false;
        
        this.initializeEventListeners();
        this.loadLLMProviders();
    }

    initializeEventListeners() {
        // LLM Configuration
        document.getElementById('llm-provider').addEventListener('change', this.onProviderChange.bind(this));
        document.getElementById('llm-model').addEventListener('change', this.onModelChange.bind(this));
        document.getElementById('manual-model').addEventListener('input', this.onManualModelChange.bind(this));
        document.getElementById('api-key').addEventListener('input', this.onApiKeyChange.bind(this));
        document.getElementById('configure-llm-btn').addEventListener('click', this.configureLLM.bind(this));
        document.getElementById('test-api-key-btn').addEventListener('click', this.testApiKey.bind(this));
        document.getElementById('toggle-api-key').addEventListener('click', this.toggleApiKeyVisibility.bind(this));

        // File Upload
        const fileUpload = document.getElementById('file-upload');
        const fileInput = document.getElementById('csv-files');
        
        fileUpload.addEventListener('click', () => fileInput.click());
        fileUpload.addEventListener('dragover', this.onDragOver.bind(this));
        fileUpload.addEventListener('dragleave', this.onDragLeave.bind(this));
        fileUpload.addEventListener('drop', this.onDrop.bind(this));
        fileInput.addEventListener('change', this.onFileSelect.bind(this));

        // Chat
        document.getElementById('chat-input').addEventListener('keypress', this.onChatKeyPress.bind(this));
        document.getElementById('send-button').addEventListener('click', this.sendMessage.bind(this));
        
        // Image Display
        document.getElementById('close-image-btn').addEventListener('click', this.hideImageDisplay.bind(this));
    }

    async loadLLMProviders() {
        try {
            const response = await fetch('/api/llm-providers');
            const data = await response.json();
            this.llmProviders = data.providers;
            this.populateProviderSelect();
        } catch (error) {
            this.showError('Failed to load LLM providers: ' + error.message);
        }
    }

    populateProviderSelect() {
        const select = document.getElementById('llm-provider');
        select.innerHTML = '<option value="">Choose a provider...</option>';
        
        Object.keys(this.llmProviders).forEach(key => {
            const provider = this.llmProviders[key];
            const option = document.createElement('option');
            option.value = key;
            option.textContent = provider.name;
            select.appendChild(option);
        });
    }

    onProviderChange(event) {
        const provider = event.target.value;
        const modelGroup = document.getElementById('model-group');
        const manualModelGroup = document.getElementById('manual-model-group');
        const apiKeyGroup = document.getElementById('api-key-group');
        const modelSelect = document.getElementById('llm-model');

        if (provider) {
            const providerConfig = this.llmProviders[provider];
            
            // Check if this provider requires manual model entry
            if (providerConfig.manual_model_entry) {
                // Show manual model input for providers like Ollama
                modelGroup.style.display = 'none';
                manualModelGroup.style.display = 'block';
            } else {
                // Show model selection dropdown for other providers
                modelGroup.style.display = 'block';
                manualModelGroup.style.display = 'none';
                modelSelect.innerHTML = '<option value="">Choose a model...</option>';
                providerConfig.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
            }

            // Show/hide API key input
            if (providerConfig.requires_api_key) {
                apiKeyGroup.style.display = 'block';
                document.getElementById('api-key').placeholder = `Enter your ${providerConfig.name} API key`;
            } else {
                apiKeyGroup.style.display = 'none';
            }
        } else {
            modelGroup.style.display = 'none';
            manualModelGroup.style.display = 'none';
            apiKeyGroup.style.display = 'none';
        }

        this.updateConfigureButton();
    }

    onModelChange() {
        this.updateConfigureButton();
    }

    onManualModelChange() {
        this.updateConfigureButton();
    }

    onApiKeyChange() {
        this.updateConfigureButton();
    }

    toggleApiKeyVisibility() {
        const apiKeyInput = document.getElementById('api-key');
        const toggleButton = document.getElementById('toggle-api-key');
        
        if (apiKeyInput.type === 'password') {
            apiKeyInput.type = 'text';
            toggleButton.textContent = '🙈';
            toggleButton.title = 'Hide API Key';
        } else {
            apiKeyInput.type = 'password';
            toggleButton.textContent = '👁️';
            toggleButton.title = 'Show API Key';
        }
    }

    updateConfigureButton() {
        const provider = document.getElementById('llm-provider').value;
        const model = document.getElementById('llm-model').value;
        const manualModel = document.getElementById('manual-model').value;
        const apiKey = document.getElementById('api-key').value;
        const configureBtn = document.getElementById('configure-llm-btn');
        const testBtn = document.getElementById('test-api-key-btn');

        if (provider) {
            const providerConfig = this.llmProviders[provider];
            
            // Determine which model input to use
            let selectedModel = '';
            if (providerConfig.manual_model_entry) {
                selectedModel = manualModel.trim();
            } else {
                selectedModel = model;
            }

            if (selectedModel) {
                const canConfigure = !providerConfig.requires_api_key || apiKey.trim();
                const canTest = providerConfig.requires_api_key && apiKey.trim();
                
                configureBtn.disabled = !canConfigure;
                testBtn.disabled = !canTest;
            } else {
                configureBtn.disabled = true;
                testBtn.disabled = true;
            }
        } else {
            configureBtn.disabled = true;
            testBtn.disabled = true;
        }
    }

    async testApiKey() {
        const provider = document.getElementById('llm-provider').value;
        const model = document.getElementById('llm-model').value;
        const manualModel = document.getElementById('manual-model').value;
        const apiKey = document.getElementById('api-key').value;
        const testBtn = document.getElementById('test-api-key-btn');

        // Determine which model input to use
        const providerConfig = this.llmProviders[provider];
        let selectedModel = '';
        if (providerConfig.manual_model_entry) {
            selectedModel = manualModel.trim();
        } else {
            selectedModel = model;
        }

        if (!provider || !selectedModel || !apiKey) {
            this.showMessage('Please select provider, model, and enter API key', 'error');
            return;
        }

        testBtn.disabled = true;
        testBtn.textContent = 'Testing...';

        try {
            const response = await fetch('/api/test-api-key', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    provider: provider,
                    model: selectedModel,
                    api_key: apiKey
                })
            });

            const result = await response.json();

            if (result.valid) {
                this.showMessage(`✅ ${result.message}`, 'success');
                if (result.test_response) {
                    this.showMessage(`Test response: "${result.test_response}"`, 'info');
                }
            } else {
                this.showMessage(`❌ ${result.message}`, 'error');
                if (result.error_details) {
                    console.error('API Key test error details:', result.error_details);
                }
            }
        } catch (error) {
            this.showMessage(`Error testing API key: ${error.message}`, 'error');
            console.error('API key test error:', error);
        } finally {
            testBtn.disabled = false;
            testBtn.textContent = 'Test API Key';
            this.updateConfigureButton(); // Re-enable based on current state
        }
    }

    async configureLLM() {
        const provider = document.getElementById('llm-provider').value;
        const model = document.getElementById('llm-model').value;
        const manualModel = document.getElementById('manual-model').value;
        const apiKey = document.getElementById('api-key').value;

        // Determine which model input to use
        const providerConfig = this.llmProviders[provider];
        let selectedModel = '';
        if (providerConfig.manual_model_entry) {
            selectedModel = manualModel.trim();
        } else {
            selectedModel = model;
        }

        this.showLoading();

        try {
            const response = await fetch('/api/configure-llm', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    provider: provider,
                    model: selectedModel,
                    api_key: apiKey || null
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.showSuccess(data.message);
                this.isLLMConfigured = true;
                this.updateStepStatus(1, 'completed');
                this.updateStepStatus(2, 'active');
                this.currentStep = 2;
            } else {
                this.showError(data.detail);
            }
        } catch (error) {
            this.showError('Failed to configure LLM: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    onDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    onDragLeave(event) {
        event.currentTarget.classList.remove('dragover');
    }

    onDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
        
        const files = Array.from(event.dataTransfer.files).filter(file => file.name.endsWith('.csv'));
        if (files.length > 0) {
            this.uploadFiles(files);
        } else {
            this.showError('Please drop only CSV files');
        }
    }

    onFileSelect(event) {
        const files = Array.from(event.target.files);
        if (files.length > 0) {
            this.uploadFiles(files);
        }
    }

    async uploadFiles(files) {
        if (!this.isLLMConfigured) {
            this.showError('Please configure an LLM first');
            return;
        }

        this.showLoading();

        try {
            const formData = new FormData();
            files.forEach(file => {
                formData.append('files', file);
            });

            const response = await fetch('/api/upload-csv', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showSuccess(data.message);
                this.displayUploadedFiles(data.files);
                this.areFilesUploaded = true;
                this.updateStepStatus(2, 'completed');
                this.updateStepStatus(3, 'active');
                this.currentStep = 3;
                this.enableChat();
            } else {
                this.showError(data.detail);
            }
        } catch (error) {
            this.showError('Failed to upload files: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayUploadedFiles(files) {
        const container = document.getElementById('uploaded-files');
        container.innerHTML = '';

        Object.values(files).forEach(file => {
            const fileDiv = document.createElement('div');
            fileDiv.className = 'file-info';
            
            const separatorName = this.getSeparatorName(file.separator);
            
            fileDiv.innerHTML = `
                <h4>📊 ${file.filename}</h4>
                <div class="file-stats">
                    <div class="stat">
                        <div class="stat-value">${file.shape[0]}</div>
                        <div class="stat-label">Rows</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${file.shape[1]}</div>
                        <div class="stat-label">Columns</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">${separatorName}</div>
                        <div class="stat-label">Separator</div>
                    </div>
                </div>
                <div class="data-preview">
                    <strong>Preview (first 5 rows):</strong>
                    ${this.createTable(file.head, file.columns)}
                </div>
            `;
            
            container.appendChild(fileDiv);
        });
    }

    getSeparatorName(separator) {
        const separatorNames = {
            ',': 'Comma (,)',
            ';': 'Semicolon (;)',
            '\t': 'Tab',
            '|': 'Pipe (|)',
            ' ': 'Space',
            'auto-detected': 'Auto-detected'
        };
        return separatorNames[separator] || separator;
    }

    createTable(data, columns) {
        if (!data || data.length === 0) return '<p>No data to preview</p>';

        let html = '<table><thead><tr>';
        columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        html += '</tr></thead><tbody>';

        data.forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                html += `<td>${row[col] !== null && row[col] !== undefined ? row[col] : ''}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        return html;
    }

    enableChat() {
        document.getElementById('chat-container').style.display = 'block';
        document.getElementById('chat-input').disabled = false;
        document.getElementById('send-button').disabled = false;
    }

    onChatKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();

        if (!message) return;

        // Add user message
        this.addMessage('user', message);
        input.value = '';
        
        // Disable input and show waiting state
        input.disabled = true;
        document.getElementById('send-button').disabled = true;
        this.updateSendButtonState(true);
        
        // Show typing indicator with a slight delay for better UX
        setTimeout(() => {
            this.showTypingIndicator();
        }, 300);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();

            // Hide typing indicator before showing response
            this.hideTypingIndicator();
            
            // Small delay to let the typing indicator fade out smoothly
            await new Promise(resolve => setTimeout(resolve, 200));

            if (response.ok) {
                this.addMessage('assistant', this.formatResponse(data.response));
            } else {
                this.addMessage('assistant', `Error: ${data.detail}`);
            }
        } catch (error) {
            // Hide typing indicator on error
            this.hideTypingIndicator();
            await new Promise(resolve => setTimeout(resolve, 200));
            this.addMessage('assistant', `Error: ${error.message}`);
        } finally {
            // Re-enable input and restore normal state
            input.disabled = false;
            document.getElementById('send-button').disabled = false;
            this.updateSendButtonState(false);
            input.focus();
        }
    }

    addMessage(sender, content) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        messageDiv.innerHTML = `
            <div class="message-header">${sender === 'user' ? 'You' : 'AI Assistant'}</div>
            <div>${content}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    formatResponse(response) {
        let formattedContent;
        
        switch (response.type) {
            case 'dataframe':
                formattedContent = this.createTable(response.data, response.columns);
                break;
            case 'number':
                formattedContent = `<strong>Result:</strong> ${response.data}`;
                break;
            case 'string':
                formattedContent = response.data;
                break;
            default:
                formattedContent = response.data;
                break;
        }
        
        // Check if the response contains a PNG path and display the image
        this.checkAndDisplayImage(formattedContent);
        
        return formattedContent;
    }

    checkAndDisplayImage(responseText) {
        // Convert response to string if it's not already
        const text = typeof responseText === 'string' ? responseText : String(responseText);
        
        // Regular expression to find PNG file paths in exports/charts/ directory
        const pngPathRegex = /exports\/charts\/[^\\s<>"']+\.png/gi;
        const matches = text.match(pngPathRegex);
        
        if (matches && matches.length > 0) {
            // Use the first PNG path found
            const imagePath = matches[0];
            console.log('Found PNG path:', imagePath);
            this.showImageDisplay(imagePath);
        }
    }

    showImageDisplay(imagePath) {
        const imageDisplayArea = document.getElementById('image-display-area');
        const generatedChart = document.getElementById('generated-chart');
        
        // Set the image source - prepend with '/' to make it relative to server root
        generatedChart.src = '/' + imagePath;
        
        // Show the image display area
        imageDisplayArea.style.display = 'block';
        
        // Scroll to the image area
        imageDisplayArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    hideImageDisplay() {
        const imageDisplayArea = document.getElementById('image-display-area');
        const generatedChart = document.getElementById('generated-chart');
        
        // Hide the image display area
        imageDisplayArea.style.display = 'none';
        
        // Clear the image source
        generatedChart.src = '';
    }

    showTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        const messagesContainer = document.getElementById('chat-messages');
        
        // Show the typing indicator with smooth animation
        typingIndicator.classList.add('show');
        
        // Scroll to bottom to show the typing indicator
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        
        // Hide the typing indicator with smooth animation
        typingIndicator.classList.remove('show');
    }

    updateSendButtonState(isWaiting = false) {
        const sendButton = document.getElementById('send-button');
        
        if (isWaiting) {
            sendButton.classList.add('waiting');
            sendButton.textContent = 'Waiting...';
        } else {
            sendButton.classList.remove('waiting');
            sendButton.textContent = 'Send';
        }
    }

    updateStepStatus(stepNumber, status) {
        const step = document.getElementById(`step${stepNumber}`);
        step.className = `step ${status}`;
    }

    showLoading() {
        document.getElementById('loading').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showMessage(message, type) {
        // Remove existing messages
        const existingMessages = document.querySelectorAll('.error, .success');
        existingMessages.forEach(msg => msg.remove());

        // Create new message
        const messageDiv = document.createElement('div');
        messageDiv.className = type;
        messageDiv.textContent = message;

        // Insert at the top of main content
        const mainContent = document.querySelector('.main-content');
        mainContent.insertBefore(messageDiv, mainContent.firstChild);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.remove();
            }
        }, 5000);
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatCSVApp();
});
