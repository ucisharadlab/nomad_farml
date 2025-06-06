document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selectors ---
    const candidateModelsTableBody = document.querySelector('#candidateModelsTable tbody');
    const modelManagementStatus = document.getElementById('modelManagementStatus');
    const newModelNameInput = document.getElementById('newModelName');
    const newModelCostInput = document.getElementById('newModelCost');
    const newModelTypeSelect = document.getElementById('newModelType');
    const modelParamsContainer = document.getElementById('modelParamsContainer');
    const customModelUploadContainer = document.getElementById('customModelUploadContainer');
    const customModelClassNameInput = document.getElementById('customModelClassName');
    const customModelFileInput = document.getElementById('customModelFile');
    const addModelBtn = document.getElementById('addModelBtn');

    const csvFileInput = document.getElementById('csvFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatusDiv = document.getElementById('uploadStatus');
    const initialModelPerformanceDiv = document.getElementById('initialModelPerformance');
    const modelSummaryTableBody = document.querySelector('#modelSummaryTable tbody');

    const nomadConfigSection = document.getElementById('nomad-config-section');
    const roleModelSelect = document.getElementById('roleModelSelect');
    const runNomadBtn = document.getElementById('runNomadBtn');
    const nomadRunStatusDiv = document.getElementById('nomadRunStatus');
    
    const resultsSection = document.getElementById('results-section');
    const liveStatusDiv = document.getElementById('liveStatus');
    const finalSummaryArea = document.getElementById('finalSummaryArea');
    
    // --- State Variables ---
    let eventSource = null;
    let totalEventsToProcess = 0;
    let classNamesOrderedForCM = [];
    let overallDataDistribution = {};
    let roleModelCostPerEvent = 0;
    let chartInstances = {};

    // --- Utility Functions ---
    function setStatus(element, message, isError = false, autoClear = false) {
        element.textContent = message;
        element.className = 'status-message';
        element.style.display = 'block';
        if (isError) {
            element.classList.add('error-message');
        } else {
             element.classList.remove('error-message');
        }
        if(autoClear) {
            setTimeout(() => { element.textContent = ''; element.style.display = 'none'; }, 5000);
        }
    }

    // --- Model Management UI ---
    const modelParamTemplates = {
        DecisionTreeClassifier: `
            <input type="number" class="model-param" data-param-name="max_depth" placeholder="Max Depth (integer, e.g., 5)">
            <input type="number" class="model-param" data-param-name="random_state" value="42" placeholder="Random State (integer)">`,
        RandomForestClassifier: `
            <input type="number" class="model-param" data-param-name="n_estimators" placeholder="N Estimators (e.g., 100)">
            <input type="number" class="model-param" data-param-name="max_depth" placeholder="Max Depth (integer, e.g., 10)">
            <input type="number" class="model-param" data-param-name="random_state" value="42" placeholder="Random State (integer)">`,
        KNeighborsClassifier: `<input type="number" class="model-param" data-param-name="n_neighbors" placeholder="N Neighbors (e.g., 5)">`,
        DummyClassifier: `<select class="model-param" data-param-name="strategy"><option value="uniform">uniform</option><option value="most_frequent">most_frequent</option></select>`,
        SVC: `<input type="text" class="model-param" data-param-name="C" value="1.0" placeholder="C (float, e.g., 1.0)"><input type="text" class="model-param" data-param-name="kernel" value="rbf" placeholder="Kernel (e.g., rbf)">`
        // LogisticRegression and GaussianNB have no common simple params to expose here
    };

    newModelTypeSelect.addEventListener('change', () => {
        const modelType = newModelTypeSelect.value;
        modelParamsContainer.innerHTML = modelParamTemplates[modelType] || '';
        customModelUploadContainer.style.display = modelType === 'custom' ? 'grid' : 'none';
    });

    async function fetchAndRenderModels() {
        try {
            const response = await fetch('/api/models');
            if (!response.ok) throw new Error('Failed to fetch models.');
            const models = await response.json();
            
            candidateModelsTableBody.innerHTML = '';
            models.forEach(model => {
                const row = candidateModelsTableBody.insertRow();
                row.insertCell().textContent = model.name;
                row.insertCell().textContent = model.is_custom ? `Custom (${model.type})` : model.type;
                row.insertCell().textContent = JSON.stringify(model.params);
                row.insertCell().textContent = model.cost.toFixed(2);
                const actionsCell = row.insertCell();
                const removeBtn = document.createElement('button');
                removeBtn.textContent = 'Remove';
                removeBtn.classList.add('small-btn', 'remove-btn');
                removeBtn.onclick = () => removeModel(model.name);
                actionsCell.appendChild(removeBtn);
            });
        } catch (error) {
            setStatus(modelManagementStatus, error.message, true);
        }
    }

    async function addModel() {
        addModelBtn.disabled = true;
        setStatus(modelManagementStatus, "Adding model...", false);
        const name = newModelNameInput.value;
        const cost = newModelCostInput.value;
        const type = newModelTypeSelect.value;

        if (!name || !cost || !type) {
            setStatus(modelManagementStatus, 'Name, cost, and type are required.', true, true);
            addModelBtn.disabled = false;
            return;
        }

        let modelData = { name, cost: parseFloat(cost), type, params: {}, is_custom: false };
        
        // Collect parameters for standard sklearn models
        document.querySelectorAll('.model-param').forEach(input => {
            let value = input.value;
            // Attempt to convert to number if it looks like one, but only if it's not empty
            if (value.trim() !== '' && !isNaN(value)) {
                value = Number(value);
            }
            if (value !== '' && value !== null) {
                modelData.params[input.dataset.paramName] = value;
            }
        });

        // Handle custom model file upload
        if (type === 'custom') {
            const className = customModelClassNameInput.value;
            const file = customModelFileInput.files[0];
            if (!className || !file) {
                setStatus(modelManagementStatus, 'Custom models require a class name and a .py file.', true, true);
                addModelBtn.disabled = false;
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('className', className);
            
            try {
                // First, upload the file and get back the module/class info
                const uploadResponse = await fetch('/api/upload_model_file', { method: 'POST', body: formData });
                const uploadResult = await uploadResponse.json();
                if (!uploadResponse.ok) throw new Error(uploadResult.error || 'Failed to upload custom model file.');

                // Now, add the model config with the info from the server
                modelData.is_custom = true;
                modelData.type = uploadResult.class_name; // The actual class name from the file
                modelData.module_name = uploadResult.module_name;
                modelData.class_name = uploadResult.class_name;

            } catch (error) {
                setStatus(modelManagementStatus, `Custom Model Error: ${error.message}`, true);
                addModelBtn.disabled = false;
                return;
            }
        }
        
        // Add the final model config to the backend
        try {
            const addResponse = await fetch('/api/models', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(modelData)
            });
            const addResult = await addResponse.json();
            if (!addResponse.ok) throw new Error(addResult.error || 'Failed to add model config.');
            
            setStatus(modelManagementStatus, `Model '${name}' added successfully.`, false, true);
            fetchAndRenderModels(); // Refresh table
            // Clear form
            newModelNameInput.value = '';
            newModelCostInput.value = '';
            newModelTypeSelect.value = '';
            modelParamsContainer.innerHTML = '';
            customModelUploadContainer.style.display = 'none';
            customModelClassNameInput.value = '';
            customModelFileInput.value = '';

        } catch (error) {
            setStatus(modelManagementStatus, `Config Error: ${error.message}`, true);
        } finally {
            addModelBtn.disabled = false;
        }
    }

    async function removeModel(modelName) {
        if (!window.confirm(`Are you sure you want to remove the model: ${modelName}?`)) return;

        try {
            const response = await fetch(`/api/models/${encodeURIComponent(modelName)}`, { method: 'DELETE' });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || 'Failed to remove model.');

            setStatus(modelManagementStatus, `Model '${modelName}' removed.`, false, true);
            fetchAndRenderModels();
        } catch (error) {
            setStatus(modelManagementStatus, error.message, true);
        }
    }
    
    addModelBtn.addEventListener('click', addModel);

    // --- Data Upload and Training ---
    uploadBtn.addEventListener('click', async () => {
        if (!csvFileInput.files.length) {
            setStatus(uploadStatusDiv, 'Please select a CSV file.', true, true);
            return;
        }
        const formData = new FormData();
        formData.append('file', csvFileInput.files[0]);

        setStatus(uploadStatusDiv, 'Uploading and training based on current models. This may take a moment...');
        initialModelPerformanceDiv.style.display = 'none';
        nomadConfigSection.style.display = 'none';
        resultsSection.style.display = 'none';
        uploadBtn.disabled = true;
        runNomadBtn.disabled = true;

        try {
            const response = await fetch('/api/upload_csv', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Upload failed.');

            setStatus(uploadStatusDiv, `${data.message} Detected Classes: ${data.classes_str.join(', ')}.`, false);
            classNamesOrderedForCM = data.classes_str || [];
            overallDataDistribution = data.initial_priors_str_keys || {};
            
            modelSummaryTableBody.innerHTML = '';
            roleModelSelect.innerHTML = '';
            if (data.model_summaries && data.model_summaries.length > 0) {
                data.model_summaries.forEach(model => {
                    const row = modelSummaryTableBody.insertRow();
                    row.insertCell().textContent = model.name;
                    row.insertCell().textContent = model.accuracy.toFixed(4);
                    row.insertCell().textContent = model.f1_score_weighted.toFixed(4);
                    row.insertCell().textContent = model.cost;
                    
                    if (!model.name.includes('(FAILED)')) {
                        const option = document.createElement('option');
                        option.value = model.name;
                        option.textContent = `${model.name} (Acc: ${model.accuracy.toFixed(3)}, Cost: ${model.cost})`;
                        roleModelSelect.appendChild(option);
                    }
                });
                initialModelPerformanceDiv.style.display = 'block';
                nomadConfigSection.style.display = 'block';
                runNomadBtn.disabled = false;
            } else {
                setStatus(uploadStatusDiv, 'Training complete, but no model summaries were returned.', true);
            }
             initializePhaseUI(classNamesOrderedForCM, overallDataDistribution);
             initializeAllCharts(data.candidate_model_names || [], classNamesOrderedForCM, overallDataDistribution);
        } catch (error) {
            setStatus(uploadStatusDiv, `Error: ${error.message}`, true);
        } finally {
            uploadBtn.disabled = false;
        }
    });

    // --- Workload Phase Management ---
    const phasesContainer = document.getElementById('phasesContainer');
    const addPhaseBtn = document.getElementById('addPhaseBtn');
    let phaseCounter = 0;

    function createPhaseElement(phaseIndex, initialData = {}) {
        phaseCounter++;
        const phaseId = `phase-ui-${phaseCounter}`;
        const phaseDiv = document.createElement('div');
        phaseDiv.classList.add('phase-block');
        phaseDiv.id = phaseId;

        let phaseHtml = `
            <h4>Phase ${phaseIndex + 1}</h4>
            <div class="phase-config-item">
                <label for="${phaseId}-duration">Duration (events):</label>
                <input type="number" id="${phaseId}-duration" class="phase-duration" value="${initialData.duration || 100}" min="1">
            </div>
            <div class="phase-distribution-title">Target Class Distribution (must sum to 1.0)</div>
            <div class="phase-class-probabilities">`;

        classNamesOrderedForCM.forEach(className => {
            const prob = initialData.target_distribution ? (initialData.target_distribution[className] || 0).toFixed(3) : (1 / classNamesOrderedForCM.length).toFixed(3);
            phaseHtml += `
                <div class="class-prob-item">
                    <label for="${phaseId}-${className}-prob" title="${className}">${className.length > 15 ? className.substring(0, 12) + '...' : className}:</label>
                    <input type="number" id="${phaseId}-${className}-prob" class="phase-class-prob" data-class-name="${className}" value="${prob}" min="0" max="1" step="0.01">
                </div>`;
        });

        phaseHtml += `</div><div class="phase-actions">
            <button type="button" class="small-btn set-overall-dist-btn">Set to Overall Dist.</button>
            <button type="button" class="small-btn remove-phase-btn">Remove Phase</button>
        </div>`;
        phaseDiv.innerHTML = phaseHtml;

        phaseDiv.querySelector('.remove-phase-btn').addEventListener('click', () => {
            phaseDiv.remove();
            updatePhaseHeaders();
        });
        phaseDiv.querySelector('.set-overall-dist-btn').addEventListener('click', () => {
             classNamesOrderedForCM.forEach(className => {
                const probInput = phaseDiv.querySelector(`.phase-class-prob[data-class-name="${className}"]`);
                if (probInput) {
                    probInput.value = (overallDataDistribution[className] || 0).toFixed(3);
                }
            });
        });

        return phaseDiv;
    }
    
    function updatePhaseHeaders() {
        const phaseBlocks = phasesContainer.querySelectorAll('.phase-block');
        phaseBlocks.forEach((block, index) => {
            block.querySelector('h4').textContent = `Phase ${index + 1}`;
        });
    }

    function addPhaseBlock(initialData = {}) {
        const phaseIndex = phasesContainer.children.length;
        const newPhaseElement = createPhaseElement(phaseIndex, initialData);
        phasesContainer.appendChild(newPhaseElement);
    }

    function initializePhaseUI() {
        phasesContainer.innerHTML = '';
        phaseCounter = 0;
        // Don't add default phases unless there is data
        if (classNamesOrderedForCM && classNamesOrderedForCM.length > 0) {
            addPhaseBlock({ duration: 100, target_distribution: overallDataDistribution });
        }
    }
    
    addPhaseBtn.addEventListener('click', () => addPhaseBlock());
    
    function getWorkloadPhasesFromUI() {
        const phases = [];
        const phaseBlocks = phasesContainer.querySelectorAll('.phase-block');
        let isValid = true;
        if (phaseBlocks.length === 0) return []; // An empty array means sequential mode

        phaseBlocks.forEach((block, index) => {
            const durationInput = block.querySelector('.phase-duration');
            const duration = parseInt(durationInput.value, 10);
            if (isNaN(duration) || duration <= 0) {
                setStatus(nomadRunStatusDiv, `Phase ${index + 1}: Duration must be a positive integer.`, true, true);
                isValid = false; return;
            }

            const target_distribution = {};
            let probSum = 0;
            const probInputs = block.querySelectorAll('.phase-class-prob');
            probInputs.forEach(input => {
                const prob = parseFloat(input.value);
                if (isNaN(prob) || prob < 0 || prob > 1) {
                    setStatus(nomadRunStatusDiv, `Phase ${index + 1}: Probabilities must be between 0 and 1.`, true, true);
                    isValid = false; return;
                }
                target_distribution[input.dataset.className] = prob;
                probSum += prob;
            });
            if (!isValid) return;

            if (Math.abs(probSum - 1.0) > 0.01) {
                setStatus(nomadRunStatusDiv, `Phase ${index + 1}: Probabilities must sum to 1.0 (current sum: ${probSum.toFixed(3)}).`, true, true);
                isValid = false; return;
            }
            phases.push({ duration, target_distribution });
        });
        return isValid ? phases : null;
    }


    // --- NOMAD Simulation ---
    runNomadBtn.addEventListener('click', () => {
        const workloadPhases = getWorkloadPhasesFromUI();
        if (workloadPhases === null) return; // Validation failed in the helper

        const config = {
            role_model_name: document.getElementById('roleModelSelect').value,
            epsilon: document.getElementById('epsilonInput').value,
            quality_metric_for_ec: document.getElementById('qualityMetricEC').value,
            safety_check_type: document.getElementById('safetyCheckType').value,
            adaptive_update_window: document.getElementById('adaptiveWindowInput').value,
            adaptive_beta: document.getElementById('adaptiveBetaInput').value,
            batch_size: 10,
            workload_phases: JSON.stringify(workloadPhases)
        };
        
        if (!config.role_model_name) {
            setStatus(nomadRunStatusDiv, "Please select a Role Model.", true, true);
            return;
        }

        setStatus(nomadRunStatusDiv, 'Initializing NOMAD simulation...');
        resultsSection.style.display = 'block';
        finalSummaryArea.style.display = 'none';
        document.getElementById('phaseShiftLog').style.display = 'none';
        document.getElementById('phaseShiftLog').innerHTML = '<strong>KL Divergence & Phase Shift Log:</strong><br>';
        
        runNomadBtn.disabled = true;
        uploadBtn.disabled = true;

        if (eventSource) eventSource.close();
        
        const queryParams = new URLSearchParams(config).toString();
        eventSource = new EventSource(`/api/nomad_event_stream?${queryParams}`);
        
        eventSource.onopen = () => setStatus(nomadRunStatusDiv, 'Stream connected. Receiving live updates...');
        eventSource.onerror = (err) => {
            setStatus(nomadRunStatusDiv, 'Stream connection error. The server might have terminated the process.', true);
            console.error("EventSource failed:", err);
            eventSource.close();
            runNomadBtn.disabled = false;
            uploadBtn.disabled = false;
        };
        eventSource.onmessage = handleStreamedEvent;
    });

    function handleStreamedEvent(event) {
        const data = JSON.parse(event.data);
        switch (data.type) {
            case "error":
                setStatus(liveStatusDiv, `Stream Error: ${data.message}`, true);
                eventSource.close();
                runNomadBtn.disabled = false;
                uploadBtn.disabled = false;
                break;
            case "setup":
                totalEventsToProcess = data.total_events;
                roleModelCostPerEvent = data.role_model_cost_per_event;
                setStatus(liveStatusDiv, `Setup complete. Total events to process: ${totalEventsToProcess}.`);
                initializeAllCharts(
                    Object.keys(data.exit_classes_info),
                    data.class_names_ordered,
                    data.initial_priors
                );
                if (chartInstances.accuracyTrend) {
                    chartInstances.accuracyTrend.roleModelStaticAccuracy = data.role_model_static_accuracy;
                }
                displayExitClasses(data.exit_classes_info);
                break;
            case "phase_shift":
                 document.getElementById('phaseShiftLog').style.display = 'block';
                 let klValue = data.kl_divergence_from_previous >= 0 ? data.kl_divergence_from_previous.toFixed(4) : "N/A";
                 let logEntry = `<strong>Phase ${data.phase_index + 1} Start</strong> (Duration: ${data.duration})<br/>
                                 &nbsp;&nbsp;KL Divergence from previous phase: ${klValue}<br/>`;
                 document.getElementById('phaseShiftLog').innerHTML += logEntry;
                 break;
            case "batch_update":
                setStatus(liveStatusDiv, `Processed event ${data.last_event_in_batch} / ${totalEventsToProcess}...`);
                updateAllCharts(data);
                break;
            case "prior_update":
                updatePriorChart(data.event_number, data.updated_priors);
                break;
            case "summary":
                setStatus(liveStatusDiv, 'Simulation complete. Final summary below.');
                displayFinalSummary(data);
                eventSource.close();
                runNomadBtn.disabled = false;
                uploadBtn.disabled = false;
                break;
        }
    }
    
    // --- Charting and Display (largely unchanged from original logic) ---
    // Includes: initializeAllCharts, updateAllCharts, updatePriorChart, etc.
    function initializeAllCharts(modelNames, classNames, initialPriors) {
        Object.values(chartInstances).forEach(chart => chart.destroy());
        chartInstances = {};
        const commonOptions = { responsive: true, maintainAspectRatio: false, animation: { duration: 0 } };
        
        if(document.getElementById('modelRunCountChart')) {
            chartInstances.modelRunCount = new Chart('modelRunCountChart', { type: 'bar', data: { labels: modelNames, datasets: [{ label: 'Times Executed', data: [], backgroundColor: 'rgba(54, 162, 235, 0.7)' }] }, options: {...commonOptions, scales: { y: { beginAtZero: true, ticks: { precision: 0 } }}} });
        }
        if(document.getElementById('cumulativeCostChart')) {
            chartInstances.cumulativeCost = new Chart('cumulativeCostChart', { type: 'line', data: { labels: [], datasets: [{ label: 'NOMAD Cost', data: [], borderColor: '#ff6384', tension: 0.1 }, { label: 'Role Model Cost', data: [], borderColor: '#36a2eb', borderDash: [5, 5] }] }, options: {...commonOptions, scales: { x: { type: 'linear', title: { text: 'Event #', display: true }}, y: { beginAtZero: true }}} });
        }
        if(document.getElementById('accuracyTrendChart')) {
            chartInstances.accuracyTrend = new Chart('accuracyTrendChart', { type: 'line', data: { labels: [], datasets: [{ label: 'NOMAD Accuracy', data: [], borderColor: '#4bc0c0', tension: 0.1 }, { label: 'Role Model Accuracy', data: [], borderColor: '#ffcd56', borderDash: [5, 5] }] }, options: {...commonOptions, scales: { x: { type: 'linear', title: { text: 'Event #', display: true }}, y: { min: 0, max: 1.0 }}} });
        }
        if(document.getElementById('classPriorsChart') && classNames && initialPriors) {
             const classColors = getDistinctColors(classNames.length);
             const priorDatasets = classNames.map((name, i) => ({ label: name, data: [{x: 0, y: initialPriors[name] || 0}], borderColor: classColors[i], tension: 0.1, fill: false }));
             chartInstances.classPriors = new Chart('classPriorsChart', { type: 'line', data: { datasets: priorDatasets }, options: {...commonOptions, scales: { x: { type: 'linear', title: { text: 'Event #', display: true }}, y: { min: 0, max: 1.0, title: { text: 'Probability', display: true } }}} });
        }
    }
    
    function updateAllCharts(batchData) {
        const xValue = batchData.last_event_in_batch;
        
        if(chartInstances.modelRunCount && batchData.model_run_counts_snapshot) {
             chartInstances.modelRunCount.data.datasets[0].data = chartInstances.modelRunCount.data.labels.map(label => batchData.model_run_counts_snapshot[label] || 0);
             chartInstances.modelRunCount.update('none');
        }

        addDataToLineChart(chartInstances.cumulativeCost, xValue, [batchData.cumulative_cost, roleModelCostPerEvent * xValue]);
        addDataToLineChart(chartInstances.accuracyTrend, xValue, [batchData.live_nomad_accuracy, chartInstances.accuracyTrend?.roleModelStaticAccuracy]);
    }

    function updatePriorChart(eventNumber, updatedPriors) {
         if (!chartInstances.classPriors) return;
        const chart = chartInstances.classPriors;
        chart.data.datasets.forEach(dataset => {
            dataset.data.push({x: eventNumber, y: updatedPriors[dataset.label] || 0});
        });
        chart.update('none');
    }

    function addDataToLineChart(chart, label, dataArray) {
        if (!chart) return;
        chart.data.labels.push(label);
        chart.data.datasets.forEach((dataset, index) => {
            // Avoid adding undefined data
            if (dataArray[index] !== undefined && dataArray[index] !== null) {
                dataset.data.push(dataArray[index]);
            }
        });
        chart.update('none');
    }
    
    function getDistinctColors(count) {
        const colors = [];
        for (let i = 0; i < count; i++) {
            const hue = (i * (360 / (count + 2))) % 360; // Use a better distribution
            colors.push(`hsl(${hue}, 85%, 50%)`);
        }
        return colors;
    }

    function displayExitClasses(exitClassesInfo) {
        let ecHtml = '<h4>Exit Classes per Model (Setup):</h4><ul>';
        for (const modelName in exitClassesInfo) {
            ecHtml += `<li><strong>${modelName}</strong>: {${exitClassesInfo[modelName].join(', ') || 'None'}}</li>`;
        }
        ecHtml += '</ul>';
        document.getElementById('exitClassesDisplay').innerHTML = ecHtml;
    }
    
    function displayFinalSummary(summary) {
        const { nomad_overall_metrics, role_model_performance } = summary;
        document.getElementById('nomadOverallMetrics').innerHTML = `
            <h4>NOMAD Performance</h4>
            <p><strong>Accuracy:</strong> ${(nomad_overall_metrics.accuracy || 0).toFixed(4)}</p>
            <p><strong>Avg F1:</strong> ${(nomad_overall_metrics.avg_metrics['f1-score'] || 0).toFixed(4)}</p>
            <p><strong>Avg Cost:</strong> ${(nomad_overall_metrics.average_cost || 0).toFixed(4)}</p>`;
            
        document.getElementById('roleModelComparison').innerHTML = `
            <h4>${role_model_performance.name} (Role Model)</h4>
            <p><strong>Accuracy:</strong> ${(role_model_performance.accuracy || 0).toFixed(4)}</p>
            <p><strong>Avg F1:</strong> ${(role_model_performance.avg_metrics['f1-score'] || 0).toFixed(4)}</p>
            <p><strong>Cost:</strong> ${(role_model_performance.cost || 0).toFixed(2)}</p>`;
        
        renderConfusionMatrix(document.getElementById('nomadCM'), nomad_overall_metrics.cm, classNamesOrderedForCM);
        renderConfusionMatrix(document.getElementById('roleModelCM'), role_model_performance.cm, classNamesOrderedForCM);
        
        finalSummaryArea.style.display = 'block';
    }

    function renderConfusionMatrix(element, cmData, labels) {
        if (!cmData || cmData.length === 0 || !labels || labels.length === 0) {
            element.innerHTML = "<p>Confusion Matrix not available.</p>"; return;
        }
        let table = '<table><thead><tr><th>True \\ Pred</th>';
        labels.forEach(l => table += `<th title="${l}">${l.length > 8 ? l.substring(0,6) + '..' : l}</th>`);
        table += '</tr></thead><tbody>';
        cmData.forEach((row, i) => {
            if (i < labels.length) { // Ensure we don't go out of bounds
                table += `<tr><td title="${labels[i]}"><strong>${labels[i].length > 8 ? labels[i].substring(0,6) + '..' : labels[i]}</strong></td>`;
                row.forEach(cell => table += `<td>${cell}</td>`);
                table += '</tr>';
            }
        });
        table += '</tbody></table>';
        element.innerHTML = table;
    }

    // --- Initial Load ---
    fetchAndRenderModels();
});
