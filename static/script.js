document.addEventListener('DOMContentLoaded', () => {
    const csvFileInput = document.getElementById('csvFile');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatusDiv = document.getElementById('uploadStatus');
    const initialModelPerformanceDiv = document.getElementById('initialModelPerformance');
    const modelSummaryTableBody = document.querySelector('#modelSummaryTable tbody');

    const nomadConfigSection = document.getElementById('nomad-config-section');
    const roleModelSelect = document.getElementById('roleModelSelect');
    const epsilonInput = document.getElementById('epsilonInput');
    const qualityMetricECSelect = document.getElementById('qualityMetricEC');
    const safetyCheckTypeSelect = document.getElementById('safetyCheckType');
    const adaptiveWindowInput = document.getElementById('adaptiveWindowInput');
    const adaptiveBetaInput = document.getElementById('adaptiveBetaInput');
    
    const phasesContainer = document.getElementById('phasesContainer'); 
    const addPhaseBtn = document.getElementById('addPhaseBtn'); 

    const runNomadBtn = document.getElementById('runNomadBtn');
    const nomadRunStatusDiv = document.getElementById('nomadRunStatus');
    
    const resultsSection = document.getElementById('results-section');
    const liveStatusDiv = document.getElementById('liveStatus');
    const phaseShiftLogDiv = document.getElementById('phaseShiftLog'); 
    const finalSummaryArea = document.getElementById('finalSummaryArea');
    const nomadOverallMetricsDiv = document.getElementById('nomadOverallMetrics');
    const roleModelComparisonDiv = document.getElementById('roleModelComparison');
    const exitClassesDiv = document.getElementById('exitClassesDisplay');
    const nomadCMDiv = document.getElementById('nomadCM');
    const roleModelCMDiv = document.getElementById('roleModelCM');

    let modelRunCountChart, cumulativeCostChart, accuracyTrendChart, classPriorsChart;
    let eventSource = null;
    let totalEventsToProcess = 0;
    let candidateModelNamesForCharts = [];
    let classNamesOrderedForCM = []; 
    let overallDataDistribution = {}; 
    let roleModelCostPerEvent = 0;
    let currentClassPriorColors = {};
    let phaseCounter = 0; // For unique IDs for phase elements

    function setStatus(element, message, isError = false) {
        element.textContent = message;
        element.className = 'status-message'; 
        if (isError) {
            element.classList.add('error-message');
        }
    }

    function getDistinctColors(count) {
        const colors = [];
        for (let i = 0; i < count; i++) {
            const hue = (i * (360 / (count * 1.2 + 1))) % 360; 
            colors.push(`hsl(${hue}, 70%, 50%)`);
        }
        return colors;
    }

    function createPhaseElement(phaseIndex, initialPhaseData = null) {
        phaseCounter++;
        const phaseId = `phase-ui-${phaseCounter}`; // More unique ID for HTML elements
        const phaseDiv = document.createElement('div');
        phaseDiv.classList.add('phase-block');
        phaseDiv.setAttribute('id', phaseId);
        phaseDiv.setAttribute('data-internal-phase-index', phaseIndex); // Keep track of logical index

        let phaseHtml = `<h4>Phase ${phaseIndex + 1}</h4>`;
        phaseHtml += `<div class="phase-config-item">
                        <label for="${phaseId}-duration">Duration (events):</label>
                        <input type="number" id="${phaseId}-duration" class="phase-duration" value="${initialPhaseData?.duration || 100}" min="1">
                      </div>`;
        
        phaseHtml += `<div class="phase-distribution-title">Target Class Distribution (must sum to 1):</div>`;
        phaseHtml += `<div class="phase-class-probabilities">`;
        
        const numClasses = classNamesOrderedForCM.length > 0 ? classNamesOrderedForCM.length : 1;
        const defaultProbForUI = (1 / numClasses).toFixed(3);

        classNamesOrderedForCM.forEach(className => {
            let initialProbVal = defaultProbForUI;
            if (initialPhaseData && initialPhaseData.target_distribution && initialPhaseData.target_distribution[className] !== undefined) {
                initialProbVal = parseFloat(initialPhaseData.target_distribution[className]).toFixed(3);
            } else if (initialPhaseData === null && overallDataDistribution[className] !== undefined && phaseIndex === 0) { 
                initialProbVal = parseFloat(overallDataDistribution[className]).toFixed(3);
            }

            phaseHtml += `<div class="class-prob-item">
                            <label for="${phaseId}-${className}-prob" title="${className}">${className.length > 15 ? className.substring(0,12)+'...' : className}:</label>
                            <input type="number" id="${phaseId}-${className}-prob" class="phase-class-prob" data-class-name="${className}" value="${initialProbVal}" min="0" max="1" step="0.01">
                          </div>`;
        });
        phaseHtml += `</div>`; 

        phaseHtml += `<div class="phase-actions">
                        <button class="small-btn set-overall-dist-btn">Set to Overall Distribution</button>
                        <button class="small-btn remove-phase-btn">Remove Phase</button>
                      </div>`;
        phaseDiv.innerHTML = phaseHtml;

        phaseDiv.querySelector('.set-overall-dist-btn').addEventListener('click', () => {
            classNamesOrderedForCM.forEach(className => {
                const probInput = phaseDiv.querySelector(`.phase-class-prob[data-class-name="${className}"]`);
                if (probInput) {
                    probInput.value = (overallDataDistribution[className] || 0).toFixed(3);
                }
            });
        });

        phaseDiv.querySelector('.remove-phase-btn').addEventListener('click', () => {
            phaseDiv.remove();
            updatePhaseIndices();
        });
        
        return phaseDiv;
    }
    
    function updatePhaseIndices() {
        const phaseBlocks = phasesContainer.querySelectorAll('.phase-block');
        phaseBlocks.forEach((block, index) => {
            block.setAttribute('data-internal-phase-index', index); // Update logical index
            block.querySelector('h4').textContent = `Phase ${index + 1}`;
        });
    }

    function addPhaseBlock(initialData = null) {
        const phaseIndex = phasesContainer.children.length;
        const newPhaseElement = createPhaseElement(phaseIndex, initialData);
        phasesContainer.appendChild(newPhaseElement);
        // updatePhaseIndices(); // No need to call here as createPhaseElement uses phaseIndex which is current children.length
    }

    addPhaseBtn.addEventListener('click', () => addPhaseBlock());

    function getWorkloadPhasesFromUI() {
        console.log("JS getWorkloadPhasesFromUI: Reading phase definitions from UI.");
        const phases = [];
        const phaseBlocks = phasesContainer.querySelectorAll('.phase-block');
        let isValid = true;

        if (phaseBlocks.length === 0) {
            setStatus(nomadRunStatusDiv, "No workload phases defined. Running default sequential simulation on test data.", false);
            console.log("JS getWorkloadPhasesFromUI: No phase blocks found, returning empty array for sequential mode.");
            return []; 
        }

        phaseBlocks.forEach((block, index) => {
            if (!isValid) return; 

            const durationInput = block.querySelector('.phase-duration');
            const duration = parseInt(durationInput.value);
            console.log(`JS getWorkloadPhasesFromUI: Phase ${index + 1}, raw duration input: '${durationInput.value}', parsed: ${duration}`);
            if (isNaN(duration) || duration <= 0 || !Number.isInteger(duration)) {
                setStatus(nomadRunStatusDiv, `Phase ${index + 1}: Duration must be a positive integer. Found: '${durationInput.value}'`, true);
                durationInput.style.borderColor = 'red';
                isValid = false; return;
            } else {
                 durationInput.style.borderColor = '';
            }

            const target_distribution = {};
            let probSum = 0;
            const probInputs = block.querySelectorAll('.phase-class-prob');
            const classNamesInData = new Set(classNamesOrderedForCM);

            probInputs.forEach(input => {
                if (!isValid) return;
                const className = input.getAttribute('data-class-name');
                const prob = parseFloat(input.value);
                console.log(`JS getWorkloadPhasesFromUI: Phase ${index + 1}, Class '${className}', raw prob input: '${input.value}', parsed: ${prob}`);

                if(!classNamesInData.has(className)) { 
                    setStatus(nomadRunStatusDiv, `Phase ${index + 1}: Unknown class name '${className}' in UI. Available: ${classNamesOrderedForCM.join(', ')}`, true);
                    isValid = false; return;
                }
                if (isNaN(prob) || prob < 0 || prob > 1) {
                    setStatus(nomadRunStatusDiv, `Phase ${index + 1}: Probability for ${className} must be between 0 and 1. Found: '${input.value}'`, true);
                    input.style.borderColor = 'red';
                    isValid = false; return;
                } else {
                    input.style.borderColor = '';
                }
                target_distribution[className] = prob;
                probSum += prob;
            });

            if (!isValid) return;

            console.log(`JS getWorkloadPhasesFromUI: Phase ${index + 1}, calculated probSum: ${probSum}`);
            if (Math.abs(probSum - 1.0) > 0.01 && Object.keys(target_distribution).length > 0) {
                setStatus(nomadRunStatusDiv, `Phase ${index + 1}: Probabilities must sum to 1.0 (current sum for specified classes: ${probSum.toFixed(3)}).`, true);
                probInputs.forEach(input => input.style.borderColor = 'red');
                isValid = false; return;
            } else {
                 probInputs.forEach(input => input.style.borderColor = '');
            }
            phases.push({ duration, target_distribution });
        });
        console.log("JS getWorkloadPhasesFromUI: Final phases collected:", JSON.stringify(phases), "Is Valid:", isValid);
        return isValid ? phases : null;
    }
    
    function initializePhaseUI() {
        console.log("JS initializePhaseUI: Setting up phase UI.");
        phasesContainer.innerHTML = ''; 
        phaseCounter = 0; // Reset counter for unique element IDs
        let defaultPhaseData = null;
        if (classNamesOrderedForCM.length > 0) {
            defaultPhaseData = {
                duration: 100, 
                target_distribution: { ...overallDataDistribution } 
            };
            classNamesOrderedForCM.forEach(cn => {
                if (defaultPhaseData.target_distribution[cn] === undefined) {
                    defaultPhaseData.target_distribution[cn] = 0.0;
                }
            });
            console.log("JS initializePhaseUI: Created default phase data:", JSON.stringify(defaultPhaseData));
        } else {
            console.log("JS initializePhaseUI: No class names available to create a default phase.");
        }
        addPhaseBlock(defaultPhaseData); 
        if(classNamesOrderedForCM.length > 1) { // Add a second contrasting phase if possible
            let secondPhaseDist = {};
            const firstClass = classNamesOrderedForCM[0];
            const secondClass = classNamesOrderedForCM[1];
            secondPhaseDist[firstClass] = 0.2;
            secondPhaseDist[secondClass] = 0.8;
            // Distribute remaining probability (0) among other classes if they exist
            for(let i=2; i<classNamesOrderedForCM.length; i++) {
                secondPhaseDist[classNamesOrderedForCM[i]] = 0.0;
            }
            addPhaseBlock({duration: 100, target_distribution: secondPhaseDist});
        }

    }

    uploadBtn.addEventListener('click', async () => {
        console.log("JS uploadBtn: Clicked.");
        if (!csvFileInput.files.length) { setStatus(uploadStatusDiv, 'Please select a CSV file.', true); return; }
        const formData = new FormData(); formData.append('file', csvFileInput.files[0]);
        setStatus(uploadStatusDiv, 'Uploading and training base models...');
        initialModelPerformanceDiv.style.display = 'none'; modelSummaryTableBody.innerHTML = ''; roleModelSelect.innerHTML = '';
        nomadConfigSection.style.display = 'none'; resultsSection.style.display = 'none'; finalSummaryArea.style.display = 'none';
        uploadBtn.disabled = true; runNomadBtn.disabled = true;

        try {
            const response = await fetch('/api/upload_csv', { method: 'POST', body: formData });
            const data = await response.json();
            console.log("JS uploadBtn: Response from /api/upload_csv:", data);
            if (!response.ok) { setStatus(uploadStatusDiv, `Error: ${data.error || 'Upload failed'}`, true); uploadBtn.disabled = false; return; }

            setStatus(uploadStatusDiv, `${data.message} Detected Classes: ${data.classes_str.join(', ')}.`);
            candidateModelNamesForCharts = data.candidate_model_names || [];
            classNamesOrderedForCM = data.classes_str || []; 
            overallDataDistribution = data.initial_priors_str_keys || {};
            console.log("JS uploadBtn: Stored overallDataDistribution:", JSON.stringify(overallDataDistribution));
            console.log("JS uploadBtn: Stored classNamesOrderedForCM:", classNamesOrderedForCM);


            const classColors = getDistinctColors(classNamesOrderedForCM.length);
            currentClassPriorColors = {};
            classNamesOrderedForCM.forEach((name, index) => { currentClassPriorColors[name] = classColors[index % classColors.length]; });

            if (data.model_summaries && data.model_summaries.length > 0) {
                data.model_summaries.forEach(model => {
                    const row = modelSummaryTableBody.insertRow();
                    row.insertCell().textContent = model.name; row.insertCell().textContent = model.accuracy.toFixed(4);
                    row.insertCell().textContent = model.f1_score_weighted.toFixed(4); row.insertCell().textContent = model.cost;
                    const option = document.createElement('option'); option.value = model.name;
                    option.textContent = `${model.name} (Acc: ${model.accuracy.toFixed(3)}, Cost: ${model.cost})`;
                    roleModelSelect.appendChild(option);
                });
                initialModelPerformanceDiv.style.display = 'block'; nomadConfigSection.style.display = 'block'; runNomadBtn.disabled = false;
            } else { setStatus(uploadStatusDiv, 'Base models trained, but no summaries returned or no models available.', true); }
            
            initializePhaseUI();
            initializeLiveCharts(candidateModelNamesForCharts, classNamesOrderedForCM, overallDataDistribution);

        } catch (error) { setStatus(uploadStatusDiv, `Client-side error during upload processing: ${error.message}`, true); console.error("Upload processing error:", error);
        } finally { uploadBtn.disabled = false; }
    });

    runNomadBtn.addEventListener('click', () => {
        console.log("JS runNomadBtn: Clicked.");
        const workloadPhases = getWorkloadPhasesFromUI(); 
        if (workloadPhases === null) { 
            console.log("JS runNomadBtn: Workload phases validation failed.");
            return; 
        }
        console.log("JS runNomadBtn: Workload phases for simulation:", JSON.stringify(workloadPhases));


        const nomadConfig = {
            role_model_name: roleModelSelect.value, epsilon: epsilonInput.value,
            quality_metric_for_ec: qualityMetricECSelect.value, safety_check_type: safetyCheckTypeSelect.value,
            batch_size: 10, adaptive_update_window: adaptiveWindowInput.value, adaptive_beta: adaptiveBetaInput.value,
            workload_phases: JSON.stringify(workloadPhases) 
        };
        console.log("JS runNomadBtn: Nomad config for API call:", nomadConfig);


        if (!nomadConfig.role_model_name) { setStatus(nomadRunStatusDiv, 'Please select a Role Model.', true); return; }
        if (parseInt(nomadConfig.adaptive_update_window) < 0) { setStatus(nomadRunStatusDiv, 'Adaptive Update Window cannot be negative.', true); return; }
        const beta = parseFloat(nomadConfig.adaptive_beta);
        if (beta < 0.0 || beta > 1.0) { setStatus(nomadRunStatusDiv, 'Adaptive Beta must be between 0.0 and 1.0.', true); return; }

        setStatus(nomadRunStatusDiv, 'Initializing NOMAD simulation stream...');
        resultsSection.style.display = 'block'; finalSummaryArea.style.display = 'none';
        liveStatusDiv.textContent = 'Preparing live stream...';
        phaseShiftLogDiv.innerHTML = '<strong>KL Divergence Log:</strong><br>'; phaseShiftLogDiv.style.display = 'none'; 

        runNomadBtn.disabled = true; uploadBtn.disabled = true;
        console.log("JS runNomadBtn: Re-initializing live charts.");
        initializeLiveCharts(candidateModelNamesForCharts, classNamesOrderedForCM, overallDataDistribution); 

        if (eventSource) {
            console.log("JS runNomadBtn: Closing existing EventSource.");
            eventSource.close();
        }
        const queryParams = new URLSearchParams(nomadConfig).toString();
        console.log("JS runNomadBtn: Connecting to EventSource with params:", queryParams);
        eventSource = new EventSource(`/api/nomad_event_stream?${queryParams}`);
        setStatus(nomadRunStatusDiv, 'Connecting to NOMAD simulation stream...');

        eventSource.onopen = () => { 
            console.log("JS EventSource: Connection opened.");
            setStatus(nomadRunStatusDiv, 'Stream connected. Receiving live updates...'); 
        };

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("JS EventSource: Message received:", data);


            if (data.type === "error") { 
                console.error("JS EventSource: Error message received:", data.message);
                setStatus(liveStatusDiv, `Stream Error: ${data.message}`, true); setStatus(nomadRunStatusDiv, 'Stream ended with error.', true);
                eventSource.close(); runNomadBtn.disabled = false; uploadBtn.disabled = false; return;
            }
            
            if (data.type === "setup") {
                console.log("JS EventSource: Setup data received:", data);
                totalEventsToProcess = data.total_events;
                roleModelCostPerEvent = data.role_model_cost_per_event || 0;
                setStatus(liveStatusDiv, `Setup complete. Total events: ${totalEventsToProcess}. Role Acc: ${data.role_model_static_accuracy.toFixed(3)}, Role Cost/Event: ${roleModelCostPerEvent.toFixed(2)}. Updates batched.`);
                if(data.exit_classes_info) displayExitClasses(data.exit_classes_info);
                phaseShiftLogDiv.innerHTML = '<strong>KL Divergence Log:</strong><br>'; phaseShiftLogDiv.style.display = 'none';

                if (classPriorsChart && data.initial_priors) {
                    classPriorsChart.data.labels = [0]; 
                    classNamesOrderedForCM.forEach(className => {
                        const dataset = classPriorsChart.data.datasets.find(ds => ds.label === className);
                        if (dataset) { dataset.data = [data.initial_priors[className] || 0]; }
                    });
                    classPriorsChart.update('none');
                }
                if (accuracyTrendChart && data.role_model_static_accuracy !== undefined) {
                    accuracyTrendChart.roleModelStaticAccuracy = data.role_model_static_accuracy;
                    accuracyTrendChart.data.datasets[1].data = []; 
                    if (accuracyTrendChart.data.labels.length > 0) { 
                        accuracyTrendChart.data.datasets[1].data = accuracyTrendChart.data.labels.map(() => accuracyTrendChart.roleModelStaticAccuracy);
                    }
                    accuracyTrendChart.update('none');
                }
            } else if (data.type === "batch_update") { 
                console.log("JS EventSource: Batch_update data for charts:", JSON.stringify(data));
                setStatus(liveStatusDiv, `Processed up to event ${data.last_event_in_batch} / ${totalEventsToProcess}...`);
                updateLiveChartsFromBatch(data); 
            } else if (data.type === "prior_update") { 
                console.log("JS EventSource: Prior_update data received:", data);
                if (classPriorsChart && data.updated_priors) {
                    const eventNum = data.event_number;
                    let pointIndex = classPriorsChart.data.labels.indexOf(eventNum);
                    if (pointIndex === -1) {
                        classPriorsChart.data.labels.push(eventNum);
                        classPriorsChart.data.labels.sort((a, b) => a - b); 
                        pointIndex = classPriorsChart.data.labels.indexOf(eventNum);
                    }
                    classNamesOrderedForCM.forEach(className => {
                        const dataset = classPriorsChart.data.datasets.find(ds => ds.label === className);
                        if (dataset) {
                            for(let i = 0; i < classPriorsChart.data.labels.length; i++) {
                                if(dataset.data[i] === undefined) { dataset.data[i] = (i > 0 && dataset.data[i-1] !== undefined) ? dataset.data[i-1] : 0; }
                            }
                            dataset.data[pointIndex] = data.updated_priors[className] || 0;
                        }
                    });
                    classPriorsChart.update('none');
                }
            } else if (data.type === "phase_shift") {
                console.log("JS EventSource: Phase_shift data received:", data);
                phaseShiftLogDiv.style.display = 'block';
                const klDiv = data.kl_divergence_from_previous !== -1.0 ? data.kl_divergence_from_previous.toFixed(4) : "N/A (first phase or error)";
                let logEntry = `Phase ${data.phase_index + 1} (Duration: ${data.duration}):<br>`;
                logEntry += `&nbsp;&nbsp;Target Dist: ${JSON.stringify(data.target_distribution)}<br>`;
                if (data.phase_index === 0) { logEntry += `&nbsp;&nbsp;Prev Dist: Initial Priors (from training data)<br>`;
                } else if (data.previous_target_distribution) { logEntry += `&nbsp;&nbsp;Prev Target Dist: ${JSON.stringify(data.previous_target_distribution)}<br>`; }
                logEntry += `&nbsp;&nbsp;KL Divergence (Q_current || P_previous): ${klDiv}<br><hr>`;
                phaseShiftLogDiv.innerHTML += logEntry; phaseShiftLogDiv.scrollTop = phaseShiftLogDiv.scrollHeight;
            } else if (data.type === "summary") {
                console.log("JS EventSource: Summary data received:", data);
                setStatus(liveStatusDiv, 'Simulation complete. Displaying final summary.');
                setStatus(nomadRunStatusDiv, 'Simulation complete. Final summary received.');
                displayFinalSummary(data); finalSummaryArea.style.display = 'block';
                eventSource.close(); runNomadBtn.disabled = false; uploadBtn.disabled = false;
            }
        };
        eventSource.onerror = (err) => { 
            console.error("JS EventSource: Connection error:", err);
            setStatus(liveStatusDiv, 'Stream connection error or ended abruptly.', true); setStatus(nomadRunStatusDiv, 'Stream error.', true);
            eventSource.close(); runNomadBtn.disabled = false; uploadBtn.disabled = false;
        };
    });

    function initializeLiveCharts(modelNames, classNames, initialPriorsForChart = {}) {
        console.log("JS initializeLiveCharts: Initializing charts. Model Names:", modelNames, "Class Names:", classNames, "Initial Priors for Chart:", JSON.stringify(initialPriorsForChart));
        const chartIds = ['modelRunCountChart', 'cumulativeCostChart', 'accuracyTrendChart', 'classPriorsChart'];
        chartIds.forEach(id => { const chartInstance = Chart.getChart(id); if (chartInstance) chartInstance.destroy(); });
        
        const commonChartOptions = { responsive: true, maintainAspectRatio: false, animation: { duration: 0 } };
        
        modelRunCountChart = new Chart(document.getElementById('modelRunCountChart').getContext('2d'), { type: 'bar', data: { labels: modelNames, datasets: [{ label: 'Times Model Executed', data: Array(modelNames.length).fill(0), backgroundColor: 'rgba(54, 162, 235, 0.7)' }] }, options: { ...commonChartOptions, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } } });
        cumulativeCostChart = new Chart(document.getElementById('cumulativeCostChart').getContext('2d'), { type: 'line', data: { labels: [], datasets: [ { label: 'NOMAD Cumulative Cost', data: [], borderColor: 'rgba(255, 99, 132, 1)', backgroundColor: 'rgba(255, 99, 132, 0.1)', fill: true, tension: 0.1 }, { label: 'Role Model Cumulative Cost', data: [], borderColor: 'rgba(54, 162, 235, 1)', backgroundColor: 'rgba(54, 162, 235, 0.1)', borderDash: [5, 5], fill: true, tension: 0.1 } ] }, options: { ...commonChartOptions, scales: { x: { type:'linear', title: { display: true, text: 'Event #' } }, y: { beginAtZero: true, title: { display: true, text: 'Cumulative Cost' } } } } });
        accuracyTrendChart = new Chart(document.getElementById('accuracyTrendChart').getContext('2d'), { type: 'line', data: { labels: [], datasets: [ { label: 'NOMAD Accuracy', data: [], borderColor: 'rgba(75, 192, 192, 1)', backgroundColor: 'rgba(75, 192, 192, 0.1)', fill:true, tension: 0.1 }, { label: 'Role Model Accuracy (Static)', data: [], borderColor: 'rgba(255, 206, 86, 1)', borderDash: [5, 5], fill: false, tension: 0.1 } ] }, options: { ...commonChartOptions, scales: { x: { type:'linear', title: { display: true, text: 'Event #' } }, y: { beginAtZero: false, min:0, max: 1.0, title: { display: true, text: 'Accuracy' } } } } });
        
        const priorDatasets = classNames.map(className => ({ 
            label: className, 
            data: (initialPriorsForChart && initialPriorsForChart[className] !== undefined) ? [initialPriorsForChart[className]] : [0], 
            borderColor: currentClassPriorColors[className] || getDistinctColors(1)[0], 
            fill: false, tension: 0.1 
        }));
        
        classPriorsChart = new Chart(document.getElementById('classPriorsChart').getContext('2d'), { 
            type: 'line', 
            data: { 
                labels: (initialPriorsForChart && Object.keys(initialPriorsForChart).length > 0 && classNames.some(cn => initialPriorsForChart[cn] !== undefined)) ? [0] : [], 
                datasets: priorDatasets 
            }, 
            options: { ...commonChartOptions, scales: { x: { type:'linear', title: { display: true, text: 'Event #' } }, y: { beginAtZero: true, min: 0, max: 1.0, title: { display: true, text: 'Prior Probability' } } }, 
                       plugins: { legend: { position: 'top', labels:{ boxWidth:15, padding:10 } } } } 
        });
        console.log("JS initializeLiveCharts: Charts initialized.");
    }

    // Using the simplified updateLineChart from previous discussion
    function updateLineChart(chart, datasetIndex, yValue, xValue, isStaticLine = false, staticValue = null) {
        if (!chart) {
            console.error("JS updateLineChart: Chart object is null/undefined. Chart ID if available:", chart ? chart.canvas.id : "Unknown");
            return;
        }
        console.log(`JS updateLineChart: ChartID=${chart.canvas.id}, DatasetIdx=${datasetIndex}, X=${xValue}, Y=${yValue}, IsStatic=${isStaticLine}, StaticVal=${staticValue}`);

        let currentXLabels = chart.data.labels;
        let targetDataset = chart.data.datasets[datasetIndex];

        if (!targetDataset) {
            console.error(`JS updateLineChart: Target dataset at index ${datasetIndex} is undefined for chart ${chart.canvas.id}. Datasets available: ${chart.data.datasets.length}`);
            return;
        }
        
        console.log("JS updateLineChart: Before update - Labels:", JSON.stringify(currentXLabels), `Target Dataset (${targetDataset.label}) Data:`, JSON.stringify(targetDataset.data));

        let pointIndex = currentXLabels.indexOf(xValue);

        if (pointIndex === -1) { // New xValue
            currentXLabels.push(xValue);
            // Ensure labels are sorted if new xValue isn't the max (though for live stream it should be)
            currentXLabels.sort((a, b) => a - b);
            pointIndex = currentXLabels.indexOf(xValue);

            // New xValue means all datasets need this point.
            // We insert a placeholder, which will be overwritten for targetDataset.
            // For others, it's usually the previous value carried forward.
            chart.data.datasets.forEach(ds => {
                const prevValue = (pointIndex > 0 && ds.data[pointIndex - 1] !== undefined) ? ds.data[pointIndex - 1] : (ds.data.length > 0 ? ds.data[0] : 0) ;
                ds.data.splice(pointIndex, 0, prevValue); // Insert value at new sorted position
            });
        }
        
        // Set the yValue for the targetDataset at the correct pointIndex
        targetDataset.data[pointIndex] = yValue;

        // For static lines, ensure all its points reflect the static value for the current labels
        if (isStaticLine && staticValue !== null) {
            chart.data.datasets[datasetIndex].data = currentXLabels.map(() => staticValue);
        }
        
        console.log("JS updateLineChart: After update - Labels:", JSON.stringify(currentXLabels), `Target Dataset (${targetDataset.label}) Data:`, JSON.stringify(targetDataset.data));
        chart.update('none'); 
    }


    function updateLiveChartsFromBatch(batchData) { 
        // console.log("JS updateLiveChartsFromBatch: Received data ->", JSON.stringify(batchData)); // Logged by caller now

        if (!batchData || batchData.type !== "batch_update") return;
        const currentEventIndexForXAxis = batchData.last_event_in_batch;

        if (modelRunCountChart && batchData.model_run_counts_snapshot) {
            const counts = batchData.model_run_counts_snapshot;
            modelRunCountChart.data.labels.forEach((label, index) => {
                modelRunCountChart.data.datasets[0].data[index] = counts[label] || 0;
            });
            modelRunCountChart.update('none');
        }
        
        if (cumulativeCostChart) {
            console.log(`JS updateLiveChartsFromBatch: Updating cumulativeCostChart. X=${currentEventIndexForXAxis}, NOMAD Y=${batchData.cumulative_cost}`);
            updateLineChart(cumulativeCostChart, 0, batchData.cumulative_cost, currentEventIndexForXAxis);
            if (roleModelCostPerEvent >= 0) {
                updateLineChart(cumulativeCostChart, 1, roleModelCostPerEvent * currentEventIndexForXAxis, currentEventIndexForXAxis);
            }
        }

        if (accuracyTrendChart) {
            console.log(`JS updateLiveChartsFromBatch: Updating accuracyTrendChart. X=${currentEventIndexForXAxis}, NOMAD Y=${batchData.live_nomad_accuracy}`);
            updateLineChart(accuracyTrendChart, 0, batchData.live_nomad_accuracy, currentEventIndexForXAxis);
            if (accuracyTrendChart.roleModelStaticAccuracy !== undefined) {
                 updateLineChart(accuracyTrendChart, 1, accuracyTrendChart.roleModelStaticAccuracy, currentEventIndexForXAxis, true, accuracyTrendChart.roleModelStaticAccuracy);
            }
        }
    }
    
    function displayExitClasses(exitClassesInfo) { 
        let ecHtml = '<h4>Exit Classes per Model (Setup):</h4><ul>';
        if (exitClassesInfo && Object.keys(exitClassesInfo).length > 0) {
            for (const modelName in exitClassesInfo) { ecHtml += `<li><strong>${modelName}</strong>: {${exitClassesInfo[modelName].join(', ') || 'None'}}</li>`; }
        } else { ecHtml += '<li>No exit class information received.</li>'; }
        ecHtml += '</ul>'; exitClassesDiv.innerHTML = ecHtml; 
    }

    function displayFinalSummary(summaryData) { 
        console.log("JS displayFinalSummary: Received summary data:", summaryData);
        if (summaryData.nomad_overall_metrics) {
            const nom = summaryData.nomad_overall_metrics;
            nomadOverallMetricsDiv.innerHTML = `<h4>NOMAD Strategy Performance (Final)</h4> <p><strong>Overall Accuracy:</strong> ${nom.accuracy.toFixed(4)}</p> <p><strong>Weighted Avg F1-Score:</strong> ${nom.avg_metrics['f1-score'] ? nom.avg_metrics['f1-score'].toFixed(4) : 'N/A'}</p> <p><strong>Average Cost per Event:</strong> ${nom.average_cost.toFixed(4)}</p>`;
            renderConfusionMatrix(nomadCMDiv, nom.cm, classNamesOrderedForCM);
        }
        if (summaryData.role_model_performance) {
            const role = summaryData.role_model_performance;
            roleModelComparisonDiv.innerHTML = `<h4>Role Model (${role.name}) Performance (Final)</h4> <p><strong>Overall Accuracy:</strong> ${role.accuracy.toFixed(4)}</p> <p><strong>Weighted Avg F1-Score:</strong> ${role.avg_metrics['f1-score'] ? role.avg_metrics['f1-score'].toFixed(4) : 'N/A'}</p> <p><strong>Cost:</strong> ${role.cost.toFixed(2)}</p>`; 
            renderConfusionMatrix(roleModelCMDiv, role.cm, classNamesOrderedForCM);
        }
        if (summaryData.final_model_run_counts && modelRunCountChart) {
             const counts = summaryData.final_model_run_counts;
            modelRunCountChart.data.labels.forEach((label, index) => { modelRunCountChart.data.datasets[0].data[index] = counts[label] || 0; });
            modelRunCountChart.update(); 
        }
        const finalEventNum = totalEventsToProcess;
        function finalizeChart(chart, finalX, dataset0FinalYProvider, dataset1FinalYProvider, isDataset1Static = false) {
            if (!chart || finalX <= 0) return;
            let currentXLabels = chart.data.labels;
            if (!currentXLabels.includes(finalX)) { currentXLabels.push(finalX); currentXLabels.sort((a,b) => a-b); }
            const finalIndex = currentXLabels.indexOf(finalX);
            chart.data.datasets.forEach((ds, idx) => {
                for(let i = 0; i < currentXLabels.length; i++) { if(ds.data[i] === undefined) { ds.data[i] = (i > 0 && ds.data[i-1] !== undefined) ? ds.data[i-1] : 0; } }
                if (idx === 0 && dataset0FinalYProvider !== undefined) { ds.data[finalIndex] = (typeof dataset0FinalYProvider === 'function') ? dataset0FinalYProvider(finalX) : dataset0FinalYProvider; }
                if (idx === 1 && dataset1FinalYProvider !== undefined) { const val = typeof dataset1FinalYProvider === 'function' ? dataset1FinalYProvider(finalX) : dataset1FinalYProvider; if (isDataset1Static) { ds.data = chart.data.labels.map(() => val); } else { ds.data[finalIndex] = val; } }
            });
            chart.update();
        }
        if (cumulativeCostChart && summaryData.nomad_overall_metrics) { finalizeChart(cumulativeCostChart, finalEventNum, summaryData.nomad_overall_metrics.average_cost * finalEventNum, (x) => roleModelCostPerEvent * x); }
        if (accuracyTrendChart && summaryData.nomad_overall_metrics) { finalizeChart(accuracyTrendChart, finalEventNum, summaryData.nomad_overall_metrics.accuracy, accuracyTrendChart.roleModelStaticAccuracy, true); }
        if (classPriorsChart) { if (!classPriorsChart.data.labels.includes(finalEventNum) && finalEventNum > 0) { classPriorsChart.data.labels.push(finalEventNum); classPriorsChart.data.labels.sort((a,b)=>a-b); } classPriorsChart.data.datasets.forEach(dataset => { for(let i = 0; i < classPriorsChart.data.labels.length; i++) { if(dataset.data[i] === undefined) { dataset.data[i] = (i > 0 && dataset.data[i-1] !== undefined) ? dataset.data[i-1] : 0; } } }); classPriorsChart.update(); }
        console.log("JS displayFinalSummary: Summary displayed and charts finalized.");
    }
    
    function renderConfusionMatrix(element, cmData, labels) { 
        if (!cmData || cmData.length === 0 || !labels || labels.length === 0) { element.innerHTML = "<p>CM data/labels not available.</p>"; return; }
        if (!Array.isArray(cmData[0]) || cmData.length !== labels.length || cmData[0].length !== labels.length) { element.innerHTML = `<p>CM dimensions mismatch. Labels: ${labels.length}, CM: ${cmData.length}x${Array.isArray(cmData[0]) ? cmData[0].length : 'N/A'}</p>`; return; }
        let tableHtml = '<table><thead><tr><th title="Actual Class \\ Predicted Class">True \\ Pred</th>';
        labels.forEach(label => { const displayLabel = (typeof label === 'string' && label.length > 10) ? label.substring(0,7)+'...' : label; tableHtml += `<th title="${label}">${displayLabel}</th>` });
        tableHtml += '</tr></thead><tbody>';
        cmData.forEach((row, i) => { const trueLabel = labels[i]; const displayTrueLabel = (typeof trueLabel === 'string' && trueLabel.length > 10) ? trueLabel.substring(0,7)+'...' : trueLabel; tableHtml += `<tr><td title="${trueLabel}"><strong>${displayTrueLabel}</strong></td>`; row.forEach(cell => tableHtml += `<td>${cell}</td>`); tableHtml += '</tr>'; });
        tableHtml += '</tbody></table>'; element.innerHTML = tableHtml;
    }    
});