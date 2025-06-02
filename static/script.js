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
    const adaptiveWindowInput = document.getElementById('adaptiveWindowInput'); // New
    const adaptiveBetaInput = document.getElementById('adaptiveBetaInput');   // New
    const runNomadBtn = document.getElementById('runNomadBtn');
    const nomadRunStatusDiv = document.getElementById('nomadRunStatus');

    const resultsSection = document.getElementById('results-section');
    const liveStatusDiv = document.getElementById('liveStatus');
    const finalSummaryArea = document.getElementById('finalSummaryArea');
    const nomadOverallMetricsDiv = document.getElementById('nomadOverallMetrics');
    const roleModelComparisonDiv = document.getElementById('roleModelComparison');
    const exitClassesDiv = document.getElementById('exitClassesDisplay');
    const nomadCMDiv = document.getElementById('nomadCM');
    const roleModelCMDiv = document.getElementById('roleModelCM');

    let modelRunCountChart, cumulativeCostChart, accuracyTrendChart, classPriorsChart; // Added classPriorsChart
    let eventSource = null;
    let totalEventsToProcess = 0;
    let candidateModelNamesForCharts = [];
    let classNamesOrderedForCM = [];
    let roleModelCostPerEvent = 0;
    let currentClassPriorColors = {}; // To store colors for prior chart lines

    function setStatus(element, message, isError = false) {
        element.textContent = message;
        element.className = 'status-message';
        if (isError) element.classList.add('error-message');
    }
    
    // Helper function to generate distinct colors for chart lines
    function getDistinctColors(count) {
        const colors = [];
        for (let i = 0; i < count; i++) {
            const hue = (i * (360 / (count * 1.2))) % 360; // Distribute hues, avoid too similar ones
            colors.push(`hsl(${hue}, 70%, 50%)`);
        }
        return colors;
    }


    uploadBtn.addEventListener('click', async () => {
        if (!csvFileInput.files.length) {
            setStatus(uploadStatusDiv, 'Please select a CSV file.', true);
            return;
        }
        const formData = new FormData();
        formData.append('file', csvFileInput.files[0]);

        setStatus(uploadStatusDiv, 'Uploading and training base models...');
        initialModelPerformanceDiv.style.display = 'none';
        modelSummaryTableBody.innerHTML = '';
        roleModelSelect.innerHTML = '';
        nomadConfigSection.style.display = 'none';
        resultsSection.style.display = 'none';
        finalSummaryArea.style.display = 'none';
        uploadBtn.disabled = true;
        runNomadBtn.disabled = true;


        try {
            const response = await fetch('/api/upload_csv', { method: 'POST', body: formData });
            const data = await response.json();

            if (!response.ok) {
                setStatus(uploadStatusDiv, `Error: ${data.error || 'Upload failed'}`, true);
                uploadBtn.disabled = false;
                return;
            }

            setStatus(uploadStatusDiv, `${data.message} Classes: ${data.classes_str.join(', ')}.`);
            candidateModelNamesForCharts = data.candidate_model_names || [];
            classNamesOrderedForCM = data.classes_str || []; // Assuming classes_str is ordered

            // Prepare colors for class priors chart
            const classColors = getDistinctColors(classNamesOrderedForCM.length);
            currentClassPriorColors = {};
            classNamesOrderedForCM.forEach((name, index) => {
                currentClassPriorColors[name] = classColors[index % classColors.length];
            });


            if (data.model_summaries && data.model_summaries.length > 0) {
                data.model_summaries.forEach(model => {
                    const row = modelSummaryTableBody.insertRow();
                    row.insertCell().textContent = model.name;
                    row.insertCell().textContent = model.accuracy.toFixed(4);
                    row.insertCell().textContent = model.f1_score_weighted.toFixed(4);
                    row.insertCell().textContent = model.cost;

                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = `${model.name} (Acc: ${model.accuracy.toFixed(3)}, Cost: ${model.cost})`;
                    roleModelSelect.appendChild(option);
                });
                initialModelPerformanceDiv.style.display = 'block';
                nomadConfigSection.style.display = 'block';
                runNomadBtn.disabled = false;
            } else {
                 setStatus(uploadStatusDiv, 'Base models trained, but no summaries returned or no models.', true);
            }
             // Initialize classPriorsChart with initial priors if available
            if (data.initial_priors_str_keys && Object.keys(data.initial_priors_str_keys).length > 0) {
                 // Call this to set up the chart structure, data will be added in 'setup' or 'prior_update'
                initializeLiveCharts(candidateModelNamesForCharts, classNamesOrderedForCM, data.initial_priors_str_keys);
            }


        } catch (error) {
            setStatus(uploadStatusDiv, `Client-side error: ${error.message}`, true);
            console.error(error);
        } finally {
            uploadBtn.disabled = false;
        }
    });

    runNomadBtn.addEventListener('click', () => {
        const nomadConfig = {
            role_model_name: roleModelSelect.value,
            epsilon: epsilonInput.value,
            quality_metric_for_ec: qualityMetricECSelect.value,
            safety_check_type: safetyCheckTypeSelect.value,
            batch_size: 10, // Default batch size for updates.
            adaptive_update_window: adaptiveWindowInput.value, // New
            adaptive_beta: adaptiveBetaInput.value            // New
        };

        if (!nomadConfig.role_model_name) {
            setStatus(nomadRunStatusDiv, 'Please select a Role Model.', true);
            return;
        }
        if (parseInt(nomadConfig.adaptive_update_window) < 0) {
            setStatus(nomadRunStatusDiv, 'Adaptive Update Window cannot be negative.', true);
            return;
        }
        const beta = parseFloat(nomadConfig.adaptive_beta);
        if (beta < 0.0 || beta > 1.0) {
             setStatus(nomadRunStatusDiv, 'Adaptive Beta must be between 0.0 and 1.0.', true);
            return;
        }


        setStatus(nomadRunStatusDiv, 'Initializing NOMAD simulation stream...');
        resultsSection.style.display = 'block';
        finalSummaryArea.style.display = 'none';
        liveStatusDiv.textContent = 'Preparing live stream...';

        runNomadBtn.disabled = true;
        uploadBtn.disabled = true;

        // Re-initialize charts (especially classPriorsChart structure if classes changed, though less likely here)
        // Data for classPriorsChart will be populated by 'setup' and 'prior_update' events.
        initializeLiveCharts(candidateModelNamesForCharts, classNamesOrderedForCM, {}); // Pass empty initial for structure

        if (eventSource) eventSource.close();

        const queryParams = new URLSearchParams(nomadConfig).toString();
        eventSource = new EventSource(`/api/nomad_event_stream?${queryParams}`);
        setStatus(nomadRunStatusDiv, 'Connecting to NOMAD simulation stream...');

        eventSource.onopen = () => {
            setStatus(nomadRunStatusDiv, 'Stream connected. Receiving live updates...');
        };

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === "error") {
                setStatus(liveStatusDiv, `Stream Error: ${data.message}`, true);
                setStatus(nomadRunStatusDiv, 'Stream ended with error.', true);
                eventSource.close();
                runNomadBtn.disabled = false;
                uploadBtn.disabled = false;
                return;
            }

            if (data.type === "setup") {
                totalEventsToProcess = data.total_events;
                classNamesOrderedForCM = data.class_names_ordered || []; // Update if necessary
                roleModelCostPerEvent = data.role_model_cost_per_event || 0;
                setStatus(liveStatusDiv, `Setup complete. Total events: ${totalEventsToProcess}. Role Acc: ${data.role_model_static_accuracy.toFixed(3)}, Role Cost/Event: ${roleModelCostPerEvent.toFixed(2)}. Updates batched.`);
                if(data.exit_classes_info) displayExitClasses(data.exit_classes_info);

                // Initialize or update classPriorsChart with initial priors from setup
                if (classPriorsChart && data.initial_priors) {
                    classPriorsChart.data.labels = [0]; // Start at event 0 for initial state
                    classNamesOrderedForCM.forEach(className => {
                        const dataset = classPriorsChart.data.datasets.find(ds => ds.label === className);
                        if (dataset) {
                            dataset.data = [data.initial_priors[className] || 0];
                        }
                    });
                    classPriorsChart.update();
                }
                if (accuracyTrendChart && data.role_model_static_accuracy !== undefined) {
                    accuracyTrendChart.roleModelStaticAccuracy = data.role_model_static_accuracy;
                     // Clear previous run data and add first point if available
                    accuracyTrendChart.data.datasets[1].data = []; // Clear previous role model line data
                    if (accuracyTrendChart.data.labels.length > 0) { // If x-axis has points
                        accuracyTrendChart.data.datasets[1].data = accuracyTrendChart.data.labels.map(() => accuracyTrendChart.roleModelStaticAccuracy);
                    }
                }


            } else if (data.type === "batch_update") {
                setStatus(liveStatusDiv, `Processed up to event ${data.last_event_in_batch} / ${totalEventsToProcess}...`);
                updateLiveChartsFromBatch(data);
            } else if (data.type === "prior_update") { // Handle new prior update
                if (classPriorsChart && data.updated_priors) {
                    const eventNum = data.event_number;
                    if (!classPriorsChart.data.labels.includes(eventNum)) {
                         classPriorsChart.data.labels.push(eventNum);
                    }
                     // Ensure labels are sorted for correct plotting progression
                    classPriorsChart.data.labels.sort((a, b) => a - b);
                    const eventIndex = classPriorsChart.data.labels.indexOf(eventNum);


                    classNamesOrderedForCM.forEach(className => {
                        const dataset = classPriorsChart.data.datasets.find(ds => ds.label === className);
                        if (dataset) {
                             // Ensure data array is long enough
                            while (dataset.data.length < classPriorsChart.data.labels.length) {
                                dataset.data.push(dataset.data.length > 0 ? dataset.data[dataset.data.length -1] : 0); // Pad with previous or 0
                            }
                            dataset.data[eventIndex] = data.updated_priors[className] || 0;
                        }
                    });
                    classPriorsChart.update();
                }
            } else if (data.type === "summary") {
                setStatus(liveStatusDiv, 'Simulation complete. Displaying final summary.');
                setStatus(nomadRunStatusDiv, 'Simulation complete. Final summary received.');
                displayFinalSummary(data);
                finalSummaryArea.style.display = 'block';
                eventSource.close();
                runNomadBtn.disabled = false;
                uploadBtn.disabled = false;
            }
        };

        eventSource.onerror = (err) => {
            setStatus(liveStatusDiv, 'Stream connection error or ended abruptly.', true);
            setStatus(nomadRunStatusDiv, 'Stream error.', true);
            console.error("EventSource failed:", err);
            eventSource.close();
            runNomadBtn.disabled = false;
            uploadBtn.disabled = false;
        };
    });

    function initializeLiveCharts(modelNames, classNames, initialPriors = {}) {
        const chartIds = ['modelRunCountChart', 'cumulativeCostChart', 'accuracyTrendChart', 'classPriorsChart']; // Added classPriorsChart
        chartIds.forEach(id => {
            const chartInstance = Chart.getChart(id);
            if (chartInstance) chartInstance.destroy();
        });

        const commonChartOptions = { responsive: true, maintainAspectRatio: false, animation: { duration: 200 } }; // Slightly slower animation

        modelRunCountChart = new Chart(document.getElementById('modelRunCountChart').getContext('2d'), {
            type: 'bar',
            data: { labels: modelNames, datasets: [{ label: 'Times Model Executed', data: Array(modelNames.length).fill(0), backgroundColor: 'rgba(54, 162, 235, 0.7)' }] },
            options: { ...commonChartOptions, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } }
        });

        cumulativeCostChart = new Chart(document.getElementById('cumulativeCostChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'NOMAD Cumulative Cost', data: [], borderColor: 'rgba(255, 99, 132, 1)', backgroundColor: 'rgba(255, 99, 132, 0.1)', fill: true, tension: 0.1 },
                    { label: 'Role Model Cumulative Cost', data: [], borderColor: 'rgba(54, 162, 235, 1)', backgroundColor: 'rgba(54, 162, 235, 0.1)', borderDash: [5, 5], fill: true, tension: 0.1 }
                ]
            },
            options: { ...commonChartOptions, scales: { x: { title: { display: true, text: 'Event #' } }, y: { beginAtZero: true, title: { display: true, text: 'Cumulative Cost' } } } }
        });

        accuracyTrendChart = new Chart(document.getElementById('accuracyTrendChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'NOMAD Accuracy', data: [], borderColor: 'rgba(75, 192, 192, 1)', backgroundColor: 'rgba(75, 192, 192, 0.1)', fill:true, tension: 0.1 },
                    { label: 'Role Model Accuracy', data: [], borderColor: 'rgba(255, 206, 86, 1)', borderDash: [5, 5], fill: false, tension: 0.1 }
                ]
            },
            options: { ...commonChartOptions, scales: { x: { title: { display: true, text: 'Event #' } }, y: { beginAtZero: false, min:0, max: 1.0, title: { display: true, text: 'Accuracy' } } } } // min:0, max:1 for accuracy
        });

        // Initialize Class Priors Chart
        const priorDatasets = classNames.map(className => ({
            label: className,
            data: initialPriors[className] ? [initialPriors[className]] : [0], // Start with initial prior at event 0
            borderColor: currentClassPriorColors[className] || '#000000', // Use pre-generated color
            fill: false,
            tension: 0.1
        }));

        classPriorsChart = new Chart(document.getElementById('classPriorsChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: initialPriors && Object.keys(initialPriors).length > 0 ? [0] : [], // X-axis: Event #, start at 0 for initial
                datasets: priorDatasets
            },
            options: {
                ...commonChartOptions,
                scales: {
                    x: { title: { display: true, text: 'Event #' } },
                    y: { beginAtZero: true, min: 0, max: 1.0, title: { display: true, text: 'Prior Probability' } }
                },
                plugins: { legend: { position: 'top', labels:{ boxWidth:15, padding:10 } } }
            }
        });
    }

    function updateLiveChartsFromBatch(batchData) {
        if (!batchData || batchData.type !== "batch_update") return;

        const currentEventIndexForXAxis = batchData.last_event_in_batch;

        if (modelRunCountChart && batchData.model_run_counts_snapshot) {
            const counts = batchData.model_run_counts_snapshot;
            modelRunCountChart.data.labels.forEach((label, index) => {
                modelRunCountChart.data.datasets[0].data[index] = counts[label] || 0;
            });
            modelRunCountChart.update('none'); // 'none' for no animation during rapid updates
        }

        let updateXAxis = false;
        if (cumulativeCostChart && currentEventIndexForXAxis !== undefined && batchData.cumulative_cost !== undefined) {
            if (!cumulativeCostChart.data.labels.includes(currentEventIndexForXAxis)) {
                cumulativeCostChart.data.labels.push(currentEventIndexForXAxis);
                updateXAxis = true; // Signal to update x-axis for other charts if needed
            }
            const currentIdx = cumulativeCostChart.data.labels.indexOf(currentEventIndexForXAxis);
            cumulativeCostChart.data.datasets[0].data[currentIdx] = batchData.cumulative_cost;
            if (roleModelCostPerEvent >= 0) {
                cumulativeCostChart.data.datasets[1].data[currentIdx] = roleModelCostPerEvent * currentEventIndexForXAxis;
            }
            cumulativeCostChart.update('none');
        }
        
        if (accuracyTrendChart && currentEventIndexForXAxis !== undefined && batchData.live_nomad_accuracy !== undefined) {
            if (updateXAxis && !accuracyTrendChart.data.labels.includes(currentEventIndexForXAxis)) { // Align x-axis if new point added by cost chart
                 accuracyTrendChart.data.labels.push(currentEventIndexForXAxis);
            } else if (!accuracyTrendChart.data.labels.includes(currentEventIndexForXAxis)) {
                 accuracyTrendChart.data.labels.push(currentEventIndexForXAxis); // Add if not present
            }

            const currentIdxAcc = accuracyTrendChart.data.labels.indexOf(currentEventIndexForXAxis);
            accuracyTrendChart.data.datasets[0].data[currentIdxAcc] = batchData.live_nomad_accuracy;
            if (accuracyTrendChart.roleModelStaticAccuracy !== undefined) {
                accuracyTrendChart.data.datasets[1].data[currentIdxAcc] = accuracyTrendChart.roleModelStaticAccuracy;
            }
             // Pad Role Model accuracy if new points were added to X axis but not yet filled for RM
            if (accuracyTrendChart.roleModelStaticAccuracy !== undefined) {
                while(accuracyTrendChart.data.datasets[1].data.length < accuracyTrendChart.data.labels.length) {
                    accuracyTrendChart.data.datasets[1].data.push(accuracyTrendChart.roleModelStaticAccuracy);
                }
            }
            accuracyTrendChart.update('none');
        }
    }

    function displayExitClasses(exitClassesInfo) {
        let ecHtml = '<h4>Exit Classes per Model (Live Setup):</h4><ul>';
        if (exitClassesInfo && Object.keys(exitClassesInfo).length > 0) {
            for (const modelName in exitClassesInfo) {
                ecHtml += `<li><strong>${modelName}</strong>: {${exitClassesInfo[modelName].join(', ') || 'None'}}</li>`;
            }
        } else {
            ecHtml += '<li>No exit class information received.</li>';
        }
        ecHtml += '</ul>';
        exitClassesDiv.innerHTML = ecHtml;
    }

    function displayFinalSummary(summaryData) {
        if (summaryData.nomad_overall_metrics) {
            const nom = summaryData.nomad_overall_metrics;
            nomadOverallMetricsDiv.innerHTML = `
                <h4>NOMAD Strategy Performance (Final)</h4>
                <p><strong>Overall Accuracy:</strong> ${nom.accuracy.toFixed(4)}</p>
                <p><strong>Weighted Avg F1-Score:</strong> ${nom.avg_metrics['f1-score'].toFixed(4)}</p>
                <p><strong>Average Cost per Event:</strong> ${nom.average_cost.toFixed(4)}</p>`;
            renderConfusionMatrix(nomadCMDiv, nom.cm, classNamesOrderedForCM);
        }

        if (summaryData.role_model_performance) {
            const role = summaryData.role_model_performance;
            roleModelComparisonDiv.innerHTML = `
                <h4>Role Model (${role.name}) Performance (Final)</h4>
                <p><strong>Overall Accuracy:</strong> ${role.accuracy.toFixed(4)}</p>
                <p><strong>Weighted Avg F1-Score:</strong> ${role.avg_metrics['f1-score'].toFixed(4)}</p>
                <p><strong>Cost:</strong> ${role.cost.toFixed(2)}</p>`;
            renderConfusionMatrix(roleModelCMDiv, role.cm, classNamesOrderedForCM);
        }

        // Final update to charts to ensure they reflect the total number of events accurately
        if (summaryData.final_model_run_counts && modelRunCountChart) {
             const counts = summaryData.final_model_run_counts;
            modelRunCountChart.data.labels.forEach((label, index) => {
                modelRunCountChart.data.datasets[0].data[index] = counts[label] || 0;
            });
            modelRunCountChart.update(); // With animation for final display
        }
        
        // Ensure charts' x-axes go up to totalEventsToProcess
        const finalEventNum = totalEventsToProcess;

        if (cumulativeCostChart && finalEventNum > 0) {
            if (!cumulativeCostChart.data.labels.includes(finalEventNum)) {
                cumulativeCostChart.data.labels.push(finalEventNum);
            }
            const finalIdx = cumulativeCostChart.data.labels.indexOf(finalEventNum);
            if (summaryData.nomad_overall_metrics) {
                 cumulativeCostChart.data.datasets[0].data[finalIdx] = summaryData.nomad_overall_metrics.average_cost * finalEventNum;
            }
            if (roleModelCostPerEvent >= 0) {
                 cumulativeCostChart.data.datasets[1].data[finalIdx] = roleModelCostPerEvent * finalEventNum;
            }
             // Pad data arrays if they are shorter
            cumulativeCostChart.data.datasets.forEach(ds => {
                while (ds.data.length < cumulativeCostChart.data.labels.length) ds.data.push(ds.data.slice(-1)[0] || 0);
            });
            cumulativeCostChart.update();
        }

        if (accuracyTrendChart && finalEventNum > 0) {
            if (!accuracyTrendChart.data.labels.includes(finalEventNum)) {
                accuracyTrendChart.data.labels.push(finalEventNum);
            }
            const finalIdxAcc = accuracyTrendChart.data.labels.indexOf(finalEventNum);
            if (summaryData.nomad_overall_metrics) {
                accuracyTrendChart.data.datasets[0].data[finalIdxAcc] = summaryData.nomad_overall_metrics.accuracy;
            }
            if (accuracyTrendChart.roleModelStaticAccuracy !== undefined) {
                accuracyTrendChart.data.datasets[1].data[finalIdxAcc] = accuracyTrendChart.roleModelStaticAccuracy;
            }
            accuracyTrendChart.data.datasets.forEach(ds => {
                while (ds.data.length < accuracyTrendChart.data.labels.length) ds.data.push(ds.data.slice(-1)[0] || 0);
            });
            accuracyTrendChart.update();
        }
        
        if (classPriorsChart && finalEventNum > 0) {
            if (!classPriorsChart.data.labels.includes(finalEventNum)) {
                classPriorsChart.data.labels.push(finalEventNum);
            }
             classPriorsChart.data.datasets.forEach(dataset => {
                while (dataset.data.length < classPriorsChart.data.labels.length) {
                    // Pad with the last known value for a smooth line to the end
                    dataset.data.push(dataset.data.length > 0 ? dataset.data[dataset.data.length - 1] : 0);
                }
            });
            classPriorsChart.update();
        }

    }

    function renderConfusionMatrix(element, cmData, labels) {
        if (!cmData || cmData.length === 0 || !labels || labels.length === 0) {
            element.innerHTML = "<p>Confusion matrix data or labels not available.</p>";
            return;
        }
        // Verify cmData is a 2D array and matches label dimensions
        if (!Array.isArray(cmData[0]) || cmData.length !== labels.length || cmData[0].length !== labels.length) {
             element.innerHTML = `<p>Confusion matrix dimensions mismatch. Labels: ${labels.length}, CM: ${cmData.length}x${Array.isArray(cmData[0]) ? cmData[0].length : 'N/A'}</p>`;
            return;
        }

        let tableHtml = '<table><thead><tr><th>True \\ Pred</th>';
        labels.forEach(label => {
            const displayLabel = (typeof label === 'string' && label.length > 10) ? label.substring(0,7)+'...' : label;
            tableHtml += `<th>${displayLabel}</th>`
        });
        tableHtml += '</tr></thead><tbody>';

        cmData.forEach((row, i) => {
            const trueLabel = labels[i];
            const displayTrueLabel = (typeof trueLabel === 'string' && trueLabel.length > 10) ? trueLabel.substring(0,7)+'...' : trueLabel;
            tableHtml += `<tr><td><strong>${displayTrueLabel}</strong></td>`;
            row.forEach(cell => tableHtml += `<td>${cell}</td>`);
            tableHtml += '</tr>';
        });
        tableHtml += '</tbody></table>';
        element.innerHTML = tableHtml;
    }
});