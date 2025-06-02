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

    let modelRunCountChart, cumulativeCostChart, accuracyTrendChart;
    let eventSource = null;
    let totalEventsToProcess = 0;
    let candidateModelNamesForCharts = []; 
    let classNamesOrderedForCM = []; 
    let roleModelCostPerEvent = 0; 

    function setStatus(element, message, isError = false) {
        element.textContent = message;
        element.className = 'status-message';
        if (isError) element.classList.add('error-message');
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
            batch_size: 10 // Default batch size for updates. Can be made configurable.
        };

        if (!nomadConfig.role_model_name) {
            setStatus(nomadRunStatusDiv, 'Please select a Role Model.', true);
            return;
        }

        setStatus(nomadRunStatusDiv, 'Initializing NOMAD simulation stream...');
        resultsSection.style.display = 'block'; 
        finalSummaryArea.style.display = 'none'; 
        liveStatusDiv.textContent = 'Preparing live stream...'; 

        runNomadBtn.disabled = true;
        uploadBtn.disabled = true; 

        initializeLiveCharts(candidateModelNamesForCharts);

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
                classNamesOrderedForCM = data.class_names_ordered || []; 
                roleModelCostPerEvent = data.role_model_cost_per_event || 0; 
                setStatus(liveStatusDiv, `Setup complete. Total events: ${totalEventsToProcess}. Role Acc: ${data.role_model_static_accuracy.toFixed(3)}, Role Cost/Event: ${roleModelCostPerEvent.toFixed(2)}. Updates batched.`);
                if(data.exit_classes_info) displayExitClasses(data.exit_classes_info);
                if (accuracyTrendChart && data.role_model_static_accuracy !== undefined) {
                    accuracyTrendChart.roleModelStaticAccuracy = data.role_model_static_accuracy;
                    accuracyTrendChart.data.datasets[1].data = []; 
                }
            } else if (data.type === "batch_update") { 
                setStatus(liveStatusDiv, `Processed up to event ${data.last_event_in_batch} / ${totalEventsToProcess}...`);
                updateLiveChartsFromBatch(data); 
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

    function initializeLiveCharts(modelNames) {
        const chartIds = ['modelRunCountChart', 'cumulativeCostChart', 'accuracyTrendChart'];
        chartIds.forEach(id => {
            const chartInstance = Chart.getChart(id);
            if (chartInstance) chartInstance.destroy();
        });
        
        const commonChartOptions = { responsive: true, maintainAspectRatio: false, animation: { duration: 0 } };

        modelRunCountChart = new Chart(document.getElementById('modelRunCountChart').getContext('2d'), {
            type: 'bar',
            data: { labels: modelNames, datasets: [{ label: 'Times Model Executed', data: Array(modelNames.length).fill(0), backgroundColor: 'rgba(54, 162, 235, 0.7)' }] },
            options: { ...commonChartOptions, scales: { y: { beginAtZero: true, ticks: { precision: 0 } } } } // Removed stepSize: 1
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
            options: { ...commonChartOptions, scales: { x: { title: { display: true, text: 'Event #' } }, y: { beginAtZero: true, min:0, max: 1.0, title: { display: true, text: 'Accuracy' } } } }
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
            modelRunCountChart.update();
        }

        if (cumulativeCostChart && currentEventIndexForXAxis !== undefined && batchData.cumulative_cost !== undefined) {
            if (!cumulativeCostChart.data.labels.includes(currentEventIndexForXAxis)) {
                cumulativeCostChart.data.labels.push(currentEventIndexForXAxis);
                cumulativeCostChart.data.datasets[0].data.push(batchData.cumulative_cost); 
                if (roleModelCostPerEvent >= 0) { // Allow 0 cost role models
                    const roleModelCumulative = roleModelCostPerEvent * currentEventIndexForXAxis;
                    cumulativeCostChart.data.datasets[1].data.push(roleModelCumulative);
                } else {
                    cumulativeCostChart.data.datasets[1].data.push(NaN);
                }
            } else { 
                const lastIdx = cumulativeCostChart.data.labels.length - 1;
                if (lastIdx >=0 && cumulativeCostChart.data.labels[lastIdx] === currentEventIndexForXAxis) {
                    cumulativeCostChart.data.datasets[0].data[lastIdx] = batchData.cumulative_cost;
                    if (roleModelCostPerEvent >= 0) {
                        cumulativeCostChart.data.datasets[1].data[lastIdx] = roleModelCostPerEvent * currentEventIndexForXAxis;
                    }
                }
            }
            cumulativeCostChart.update();
        }

        if (accuracyTrendChart && currentEventIndexForXAxis !== undefined && batchData.live_nomad_accuracy !== undefined) {
            if (!accuracyTrendChart.data.labels.includes(currentEventIndexForXAxis)) {
                accuracyTrendChart.data.labels.push(currentEventIndexForXAxis);
                accuracyTrendChart.data.datasets[0].data.push(batchData.live_nomad_accuracy); 
                if (accuracyTrendChart.roleModelStaticAccuracy !== undefined) {
                    accuracyTrendChart.data.datasets[1].data.push(accuracyTrendChart.roleModelStaticAccuracy);
                }
            } else {
                const lastIdx = accuracyTrendChart.data.labels.length -1;
                if (lastIdx >=0 && accuracyTrendChart.data.labels[lastIdx] === currentEventIndexForXAxis) {
                     accuracyTrendChart.data.datasets[0].data[lastIdx] = batchData.live_nomad_accuracy;
                     if (accuracyTrendChart.roleModelStaticAccuracy !== undefined) {
                        accuracyTrendChart.data.datasets[1].data[lastIdx] = accuracyTrendChart.roleModelStaticAccuracy;
                     }
                }
            }
            accuracyTrendChart.update();
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
        
        if (summaryData.final_model_run_counts && modelRunCountChart) {
             const counts = summaryData.final_model_run_counts;
            modelRunCountChart.data.labels.forEach((label, index) => {
                modelRunCountChart.data.datasets[0].data[index] = counts[label] || 0;
            });
            modelRunCountChart.update();
        }

        if (cumulativeCostChart && totalEventsToProcess > 0 && roleModelCostPerEvent >= 0) {
            // Ensure the last point for role model cost is plotted correctly if it wasn't part of the last batch update
            const lastBatchEvent = cumulativeCostChart.data.labels.length > 0 ? cumulativeCostChart.data.labels[cumulativeCostChart.data.labels.length - 1] : 0;

            if (totalEventsToProcess > lastBatchEvent) { // If the stream ended before a full final batch
                cumulativeCostChart.data.labels.push(totalEventsToProcess);
                
                // Get last nomad cumulative cost (from summary or extrapolate)
                let finalNomadTotalCost = batchData.cumulative_cost; // Use last known
                if(summaryData.nomad_overall_metrics && summaryData.nomad_overall_metrics.average_cost !== undefined){
                    finalNomadTotalCost = summaryData.nomad_overall_metrics.average_cost * totalEventsToProcess;
                }
                cumulativeCostChart.data.datasets[0].data.push(finalNomadTotalCost);
                cumulativeCostChart.data.datasets[1].data.push(roleModelCostPerEvent * totalEventsToProcess);
            } else if (cumulativeCostChart.data.labels.includes(totalEventsToProcess)) { // Ensure the last point value is correct
                const finalIdx = cumulativeCostChart.data.labels.indexOf(totalEventsToProcess);
                 let finalNomadTotalCost = summaryData.nomad_overall_metrics ? summaryData.nomad_overall_metrics.average_cost * totalEventsToProcess : cumulativeCostChart.data.datasets[0].data[finalIdx];
                cumulativeCostChart.data.datasets[0].data[finalIdx] = finalNomadTotalCost;
                cumulativeCostChart.data.datasets[1].data[finalIdx] = roleModelCostPerEvent * totalEventsToProcess;
            }
            cumulativeCostChart.update();
        }
         if (accuracyTrendChart && totalEventsToProcess > 0) {
            const lastBatchEventAcc = accuracyTrendChart.data.labels.length > 0 ? accuracyTrendChart.data.labels[accuracyTrendChart.data.labels.length - 1] : 0;
             if (totalEventsToProcess > lastBatchEventAcc) {
                accuracyTrendChart.data.labels.push(totalEventsToProcess);
                let finalNomadAcc = summaryData.nomad_overall_metrics ? summaryData.nomad_overall_metrics.accuracy : accuracyTrendChart.data.datasets[0].data.slice(-1)[0];
                accuracyTrendChart.data.datasets[0].data.push(finalNomadAcc);
                if (accuracyTrendChart.roleModelStaticAccuracy !== undefined) {
                    accuracyTrendChart.data.datasets[1].data.push(accuracyTrendChart.roleModelStaticAccuracy);
                }
             } else if (accuracyTrendChart.data.labels.includes(totalEventsToProcess)) {
                const finalIdxAcc = accuracyTrendChart.data.labels.indexOf(totalEventsToProcess);
                let finalNomadAcc = summaryData.nomad_overall_metrics ? summaryData.nomad_overall_metrics.accuracy : accuracyTrendChart.data.datasets[0].data[finalIdxAcc];
                accuracyTrendChart.data.datasets[0].data[finalIdxAcc] = finalNomadAcc;
                 if (accuracyTrendChart.roleModelStaticAccuracy !== undefined) {
                    accuracyTrendChart.data.datasets[1].data[finalIdxAcc] = accuracyTrendChart.roleModelStaticAccuracy;
                 }
             }
             accuracyTrendChart.update();
         }
    }
    
    function renderConfusionMatrix(element, cmData, labels) {
        if (!cmData || !labels || labels.length === 0) {
            element.innerHTML = "<p>Confusion matrix data / labels not available.</p>";
            return;
        }
        let tableHtml = '<table><thead><tr><th>True \\ Pred</th>';
        labels.forEach(label => tableHtml += `<th>${label.length > 10 ? label.substring(0,7)+'...' : label}</th>`);
        tableHtml += '</tr></thead><tbody>';
        cmData.forEach((row, i) => {
            const trueLabelDisplay = (labels[i] && labels[i].length > 10) ? labels[i].substring(0,7)+'...' : labels[i] || "N/A";
            tableHtml += `<tr><td><strong>${trueLabelDisplay}</strong></td>`;
            row.forEach(cell => tableHtml += `<td>${cell}</td>`);
            tableHtml += '</tr>';
        });
        tableHtml += '</tbody></table>';
        element.innerHTML = tableHtml;
    }    
});