import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { ArrowUpTrayIcon, TrashIcon, PlusIcon, PlayIcon, InformationCircleIcon, SparklesIcon, ArrowPathIcon, EyeIcon, ChevronDownIcon, ChevronUpIcon, ServerStackIcon, CpuChipIcon, ShareIcon } from '@heroicons/react/24/outline';
import { CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/solid';

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend);

// --- Constants ---
const API_BASE_URL = 'http://127.0.0.1:5001';
const MODEL_PARAM_TEMPLATES = {
    DecisionTreeClassifier: [
        { name: 'max_depth', type: 'number', placeholder: 'Max Depth (e.g., 5)' },
        { name: 'random_state', type: 'number', placeholder: 'Random State', defaultValue: 42 },
    ],
    RandomForestClassifier: [
        { name: 'n_estimators', type: 'number', placeholder: 'N Estimators (e.g., 100)' },
        { name: 'max_depth', type: 'number', placeholder: 'Max Depth (e.g., 10)' },
        { name: 'random_state', type: 'number', placeholder: 'Random State', defaultValue: 42 },
    ],
    KNeighborsClassifier: [
        { name: 'n_neighbors', type: 'number', placeholder: 'N Neighbors (e.g., 5)' },
    ],
    DummyClassifier: [
        { name: 'strategy', type: 'select', options: ['uniform', 'most_frequent'] }
    ],
    SVC: [
      { name: 'C', type: 'number', placeholder: 'C (e.g., 1.0)', defaultValue: 1.0 },
      { name: 'kernel', type: 'select', options: ['rbf', 'linear', 'poly', 'sigmoid'], defaultValue: 'rbf' }
    ]
};

// --- Helper & UI Components ---
const Section = ({ title, step, children }) => (
    <section className="bg-white/50 backdrop-blur-sm border border-gray-200/50 rounded-2xl shadow-lg p-6 md:p-8 mb-8 transition-all duration-300">
        <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center">
            <span className="bg-blue-600 text-white rounded-full h-8 w-8 flex items-center justify-center mr-3 text-lg">{step}</span>
            {title}
        </h2>
        <div className="pl-11">{children}</div>
    </section>
);

const Button = ({ children, onClick, disabled = false, icon, className = '' }) => (
    <button
        onClick={onClick}
        disabled={disabled}
        className={`inline-flex items-center justify-center px-4 py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-75 transition-all duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed ${className}`}
    >
        {icon && React.createElement(icon, { className: "h-5 w-5 mr-2" })}
        {children}
    </button>
);

const Input = ({ ...props }) => (
    <input className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 transition" {...props} />
);

const Select = ({ children, ...props }) => (
  <select className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 transition bg-white" {...props}>
    {children}
  </select>
);

const StatusMessage = ({ message, type }) => {
    if (!message) return null;
    const isError = type === 'error';
    const isInfo = type === 'info';
    const Icon = isError ? XCircleIcon : (isInfo ? InformationCircleIcon : CheckCircleIcon);
    const colors = isError ? 'bg-red-100 border-red-400 text-red-700' : (isInfo ? 'bg-blue-100 border-blue-400 text-blue-700' : 'bg-green-100 border-green-400 text-green-700');
    return (
        <div className={`my-4 p-4 border rounded-lg flex items-center ${colors}`} role="alert">
            <Icon className="h-5 w-5 mr-3 flex-shrink-0" />
            <span className="font-medium">{message}</span>
        </div>
    );
};

const ChartComponent = ({ chartData, chartOptions, type }) => {
  const ChartType = type === 'line' ? Line : Bar;
  return (
      <div className="bg-white p-4 rounded-xl shadow-md h-[350px] flex flex-col">
          <h3 className="text-lg font-semibold text-gray-700 text-center mb-2">{chartOptions.plugins.title.text}</h3>
          <div className="relative flex-grow">
              <ChartType options={chartOptions} data={chartData} />
          </div>
      </div>
  );
};

// --- Application Components ---

const ModelManagement = ({ models, setModels, setStatus }) => {
    const [newModel, setNewModel] = useState({ name: '', cost: '', type: '', params: {} });
    const [customModelFile, setCustomModelFile] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const fetchModels = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models`);
            if (!response.ok) throw new Error('Failed to fetch models.');
            const data = await response.json();
            setModels(data);
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
        }
    }, [setModels, setStatus]);

    useEffect(() => {
        fetchModels();
    }, [fetchModels]);

    const handleAddModel = async () => {
        setIsLoading(true);
        setStatus({ message: '', type: 'info' });
        let modelPayload = { ...newModel, cost: parseFloat(newModel.cost) };
        
        if (newModel.type === 'custom') {
            if (!customModelFile || !newModel.params.className) {
                setStatus({ message: 'Custom model requires a .py file and a class name.', type: 'error' });
                setIsLoading(false);
                return;
            }
            const formData = new FormData();
            formData.append('file', customModelFile);
            formData.append('className', newModel.params.className);
            try {
                const uploadRes = await fetch(`${API_BASE_URL}/api/upload_model_file`, { method: 'POST', body: formData });
                const uploadData = await uploadRes.json();
                if (!uploadRes.ok) throw new Error(uploadData.error || 'Failed to upload custom model.');
                modelPayload = {...modelPayload, is_custom: true, type: uploadData.class_name, module_name: uploadData.module_name, class_name: uploadData.class_name};
            } catch (error) {
                setStatus({ message: error.message, type: 'error' });
                setIsLoading(false);
                return;
            }
        }
        try {
            const addRes = await fetch(`${API_BASE_URL}/api/models`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(modelPayload) });
            const addData = await addRes.json();
            if (!addRes.ok) throw new Error(addData.error || 'Failed to add model.');
            setStatus({ message: addData.message, type: 'success' });
            fetchModels();
            setNewModel({ name: '', cost: '', type: '', params: {} });
            setCustomModelFile(null);
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleRemoveModel = async (modelName) => {
        setStatus({ message: '', type: 'info' });
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/${encodeURIComponent(modelName)}`, { method: 'DELETE' });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Failed to delete model.' }));
                throw new Error(errorData.error);
            }
            const data = await response.json();
            setStatus({ message: data.message, type: 'success' });
            setModels(prevModels => prevModels.filter(m => m.name !== modelName));
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
        }
    };

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setNewModel(prev => ({ ...prev, [name]: value }));
    };
    
    const handleParamChange = (e) => {
        const { name, value, type } = e.target;
        setNewModel(prev => ({
            ...prev,
            params: { ...prev.params, [name]: type === 'number' && value !== '' ? Number(value) : value }
        }));
    };
    
    const handleTypeChange = (e) => {
        const type = e.target.value;
        const templateParams = MODEL_PARAM_TEMPLATES[type] || [];
        const initialParams = {};
        templateParams.forEach(param => {
            if (param.defaultValue !== undefined) initialParams[param.name] = param.defaultValue;
            else if (param.type === 'select' && param.options) initialParams[param.name] = param.options[0];
        });
        setNewModel(prev => ({ ...prev, type: type, params: initialParams }));
    };

    return (
        <Section title="Manage Candidate Models" step="1">
            <div className="overflow-x-auto bg-white rounded-lg shadow">
                <table className="w-full text-sm text-left text-gray-500">
                    <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                        <tr>
                            <th scope="col" className="px-6 py-3">Name</th>
                            <th scope="col" className="px-6 py-3">Type</th>
                            <th scope="col" className="px-6 py-3">Cost</th>
                            <th scope="col" className="px-6 py-3">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {models.map(model => (
                            <tr key={model.name} className="bg-white border-b hover:bg-gray-50">
                                <th scope="row" className="px-6 py-4 font-medium text-gray-900 whitespace-nowrap">{model.name}</th>
                                <td className="px-6 py-4">{model.is_custom ? `Custom (${model.type})` : model.type}</td>
                                <td className="px-6 py-4">{model.cost.toFixed(2)}</td>
                                <td className="px-6 py-4">
                                    <button onClick={() => handleRemoveModel(model.name)} className="text-red-600 hover:text-red-800">
                                        <TrashIcon className="h-5 w-5"/>
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="mt-6 pt-6 border-t">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Add New Model</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <Input name="name" value={newModel.name} onChange={handleInputChange} placeholder="Unique Model Name" />
                    <Input name="cost" value={newModel.cost} onChange={handleInputChange} type="number" placeholder="Cost (e.g., 5.5)" />
                    <Select name="type" value={newModel.type} onChange={handleTypeChange}>
                      <option value="">Select Model Type</option>
                      {Object.keys(MODEL_PARAM_TEMPLATES).map(t => <option key={t} value={t}>{t}</option>)}
                      <option value="custom">-- Custom Model --</option>
                    </Select>
                </div>
                {newModel.type && newModel.type !== 'custom' && MODEL_PARAM_TEMPLATES[newModel.type] && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4 p-4 bg-gray-50 rounded-lg">
                        {MODEL_PARAM_TEMPLATES[newModel.type].map(param => (
                          param.type === 'select' ?
                          <Select key={param.name} name={param.name} onChange={handleParamChange} value={newModel.params[param.name] || ''}>
                            {param.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                          </Select>
                          :
                          <Input key={param.name} name={param.name} type={param.type} placeholder={param.placeholder} onChange={handleParamChange} value={newModel.params[param.name] || ''}/>
                        ))}
                    </div>
                )}
                {newModel.type === 'custom' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4 p-4 bg-gray-50 rounded-lg">
                        <Input name="className" placeholder="Python Class Name in File" onChange={e => setNewModel(p=>({...p, params: {...p.params, className: e.target.value}}))} />
                        <Input type="file" accept=".py" onChange={(e) => setCustomModelFile(e.target.files[0])} />
                    </div>
                )}
                <Button onClick={handleAddModel} disabled={isLoading} icon={PlusIcon}>
                    {isLoading ? 'Adding...' : 'Add Model'}
                </Button>
            </div>
        </Section>
    );
};

const DataUpload = ({ setStatus, setInitialResults, setNomadConfigReady }) => {
    const [file, setFile] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };
    
    const handleUpload = async () => {
        if (!file) { setStatus({ message: 'Please select a CSV file.', type: 'error' }); return; }
        setIsLoading(true);
        setStatus({ message: 'Uploading & training...', type: 'info' });
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch(`${API_BASE_URL}/api/upload_csv`, { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Upload failed.');
            setStatus({ message: `Training complete. Detected ${data.classes_str.length} classes.`, type: 'success' });
            setInitialResults(data);
            setNomadConfigReady(true);
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
            setNomadConfigReady(false);
        } finally {
            setIsLoading(false);
        }
    };
    return (
        <Section title="Upload Data & Train" step="2">
            <p className="text-gray-600 mb-4">Upload a CSV where the second to last column is the target variable.</p>
            <div className="flex flex-col sm:flex-row items-center gap-4">
                <input
                    type="file"
                    accept=".csv"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    className="hidden" // Hide the default input
                />
                <button
                    onClick={() => fileInputRef.current.click()}
                    className="flex-grow w-full sm:w-auto px-4 py-2 border border-gray-300 rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 text-left"
                >
                    {file ? file.name : "Choose a file..."}
                </button>
                <Button onClick={handleUpload} disabled={isLoading || !file} icon={ArrowUpTrayIcon}>
                    {isLoading ? 'Processing...' : 'Upload & Train'}
                </Button>
            </div>
        </Section>
    );
};

const DataSummary = ({ initialResults }) => {
    const [data, setData] = useState({ columns: [], data: [] });

    useEffect(() => {
        if (initialResults && initialResults.data_head) {
            setData(JSON.parse(initialResults.data_head));
        }
    }, [initialResults]);

    return (
        <div className="mt-6 space-y-4">
            <div>
                <h3 className="text-lg font-semibold text-gray-700">Data Summary</h3>
                <div className="grid grid-cols-2 gap-4 mt-2 text-sm">
                    <p className="text-gray-600"><strong>Features:</strong> {initialResults.num_features}</p>
                    <p className="text-gray-600"><strong>Classes:</strong> {initialResults.classes_str.length}</p>
                </div>
            </div>
            <div>
                <h4 className="font-semibold text-gray-700 mb-2">Data Peek (First 5 Rows)</h4>
                <div className="overflow-x-auto bg-white rounded-lg shadow">
                    <table className="w-full text-sm text-left text-gray-500">
                        <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                            <tr>{data.columns.map(col => <th key={col} className="px-4 py-2">{col}</th>)}</tr>
                        </thead>
                        <tbody>
                            {data.data.map((row, i) => <tr key={i} className="bg-white border-b hover:bg-gray-50">{row.map((cell, j) => <td key={j} className="px-4 py-2">{cell}</td>)}</tr>)}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};


const InitialTrainingSummary = ({ initialResults }) => (
    <div className="mt-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Individual Model Performance</h3>
        <div className="overflow-x-auto bg-white rounded-lg shadow">
            <table className="w-full text-sm text-left text-gray-500">
                <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                    <tr><th className="px-6 py-3">Model</th><th className="px-6 py-3">Accuracy</th><th className="px-6 py-3">W. F1</th><th className="px-6 py-3">Cost</th></tr>
                </thead>
                <tbody>
                    {initialResults.model_summaries.map(model => (
                        <tr key={model.name} className="bg-white border-b hover:bg-gray-50">
                            <th className={`px-6 py-4 font-medium whitespace-nowrap ${model.name.includes('FAILED') ? 'text-red-600' : 'text-gray-900'}`}>{model.name}</th>
                            <td className="px-6 py-4">{model.accuracy.toFixed(4)}</td><td className="px-6 py-4">{model.f1_score_weighted.toFixed(4)}</td><td className="px-6 py-4">{model.cost}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
);


const NomadConfig = ({ initialResults, setSimulationConfig, setStatus }) => {
    const [config, setConfig] = useState({ role_model_name: '', epsilon: 0.05, quality_metric_for_ec: 'f1-score', safety_check_type: 'conservative', adaptive_update_window: 100, adaptive_beta: 0.3, });
    const [phases, setPhases] = useState([]);
    useEffect(() => {
        if (initialResults.model_summaries?.length > 0) {
            const bestModel = [...initialResults.model_summaries].filter(m => !m.name.includes('FAILED')).sort((a, b) => b.accuracy - a.accuracy)[0];
            setConfig(c => ({...c, role_model_name: bestModel?.name || '' }));
        }
    }, [initialResults.model_summaries]);
    const handleInputChange = (e) => setConfig(prev => ({...prev, [e.target.name]: e.target.value }));
    const handleRunSimulation = () => {
        if (!config.role_model_name) { setStatus({ message: 'Please select a Role Model.', type: 'error' }); return; }
        for (const phase of phases) {
            const sum = Object.values(phase.target_distribution).reduce((a, b) => a + b, 0);
            if (Math.abs(sum - 1.0) > 0.01) { setStatus({ message: `Probabilities in a phase must sum to 1.0.`, type: 'error' }); return; }
        }
        setSimulationConfig({ ...config, workload_phases: JSON.stringify(phases) });
    };
    const addPhase = () => setPhases(p => [...p, { duration: 100, target_distribution: initialResults.initial_priors_str_keys || {} }]);
    const removePhase = (index) => setPhases(p => p.filter((_, i) => i !== index));
    const updatePhase = (index, field, value) => setPhases(p => p.map((phase, i) => i === index ? { ...phase, [field]: value } : phase));
    const handleResetPhaseDistribution = (phaseIndex) => updatePhase(phaseIndex, 'target_distribution', initialResults.initial_priors_str_keys);
    const updatePhaseDist = (phaseIndex, className, value) => { setPhases(p => p.map((phase, i) => i === phaseIndex ? { ...phase, target_distribution: { ...phase.target_distribution, [className]: parseFloat(value) || 0 } } : phase)); };
    return (
        <Section title="Configure NOMAD Engine" step="4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              <div><label className="font-semibold text-gray-700 block mb-1">Role Model</label><Select name="role_model_name" value={config.role_model_name} onChange={handleInputChange}><option value="">Select a Role Model</option>{initialResults.model_summaries?.filter(m => !m.name.includes('FAILED')).map(m => (<option key={m.name} value={m.name}>{`${m.name} (Acc: ${m.accuracy.toFixed(3)})`}</option>))}</Select></div>
              <div><label className="font-semibold text-gray-700 block mb-1">Epsilon (ε)</label><Input type="number" name="epsilon" value={config.epsilon} onChange={handleInputChange} step="0.01" min="0" /></div>
              <div><label className="font-semibold text-gray-700 block mb-1">Quality Metric</label><Select name="quality_metric_for_ec" value={config.quality_metric_for_ec} onChange={handleInputChange}><option value="f1-score">f1-score</option><option value="precision">precision</option><option value="recall">recall</option></Select></div>
              <div><label className="font-semibold text-gray-700 block mb-1">Safety Check</label><Select name="safety_check_type" value={config.safety_check_type} onChange={handleInputChange}><option value="conservative">Conservative</option><option value="relaxed">Relaxed</option></Select></div>
              <div><label className="font-semibold text-gray-700 block mb-1">Adaptive Window</label><Input type="number" name="adaptive_update_window" value={config.adaptive_update_window} onChange={handleInputChange} min="0" /></div>
              <div><label className="font-semibold text-gray-700 block mb-1">Adaptive Beta (β)</label><Input type="number" name="adaptive_beta" value={config.adaptive_beta} onChange={handleInputChange} step="0.05" min="0" max="1" /></div>
            </div>
            <div className="mt-8 pt-6 border-t">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Define Workload Phases (Optional)</h3>
                <div className="space-y-4">
                  {phases.map((phase, i) => (
                    <div key={i} className="p-4 bg-gray-50 rounded-lg border">
                      <div className="flex justify-between items-center mb-2"><h4 className="font-bold text-gray-600">Phase {i+1}</h4><div className="flex items-center gap-3"><button onClick={() => handleResetPhaseDistribution(i)} className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"><ArrowPathIcon className="h-4 w-4" /> Reset</button><button onClick={() => removePhase(i)} className="text-red-500 hover:text-red-700"><TrashIcon className="h-5 w-5"/></button></div></div>
                      <Input type="number" value={phase.duration} onChange={e => updatePhase(i, 'duration', parseInt(e.target.value))} placeholder="Duration (events)"/>
                      <div className="mt-2 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">{Object.keys(phase.target_distribution).map(cn => (<div key={cn}><label className="text-xs text-gray-500 truncate" title={cn}>{cn}</label><Input type="number" value={phase.target_distribution[cn]} onChange={e => updatePhaseDist(i, cn, e.target.value)} step="0.01" min="0" max="1"/></div>))}</div>
                    </div>
                  ))}
                </div>
                <Button onClick={addPhase} icon={PlusIcon} className="mt-4 bg-gray-600 hover:bg-gray-700">Add Phase</Button>
            </div>
            <div className="mt-8 text-center"><Button onClick={handleRunSimulation} icon={PlayIcon} className="text-lg px-8 py-3 bg-green-600 hover:bg-green-700">Run NOMAD Simulation</Button></div>
        </Section>
    );
};

const CascadeVisualizer = ({ setStatus }) => {
    const [imageUrls, setImageUrls] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    
    const handleSelection = async (value) => {
        if (!value) return;
        setIsLoading(true);
        setImageUrls([]);
        setStatus({ message: '', type: 'info' });
        try {
            const response = await fetch(`${API_BASE_URL}/api/visualizations?selected=${value}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to fetch visualization.');
            setImageUrls(data.plots);
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Section title="Cascade Visualizer" step="3">
            <p className="text-gray-600 mb-4">Select an option from the dropdowns below to view pre-generated model visualizations.</p>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
                <Select onChange={e => handleSelection(e.target.value)} defaultValue="">
                    <option value="" disabled>Load Model</option>
                    <option value="model_option1">Instance Distributor</option>
                    <option value="model_option2">Submodel 1</option>
                    <option value="model_option3">Submodel 2</option>
                    <option value="model_option4">Submodel 3</option>
                    <option value="model_option5">Submodel 4</option>
                </Select>
                <Select onChange={e => handleSelection(e.target.value)} defaultValue="">
                    <option value="" disabled>Test Model</option>
                    <option value="test_method_option1">Generate Test Summary</option>
                    <option value="test_method_option2">Trace Decision Path</option>
                </Select>
                 <Select onChange={e => handleSelection(e.target.value)} defaultValue="">
                    <option value="" disabled>Attack Model</option>
                    <option value="attack_option1">Compute Feature Correlation</option>
                    <option value="attack_option2">Generate Attack Result</option>
                </Select>
                <Select onChange={e => handleSelection(e.target.value)} defaultValue="">
                    <option value="" disabled>Adversarial Re-training</option>
                    <option value="adtrain_option1">Generate Ad-Train Results</option>
                    <option value="adtrain_option2">Compare Decision Boundary</option>
                </Select>
            </div>
            <div className="mt-4 p-4 border rounded-lg min-h-[200px] bg-gray-50 flex items-center justify-center">
                {isLoading && <p>Loading Visualization...</p>}
                {!isLoading && imageUrls.length === 0 && <p className="text-gray-500">Select an option to view a visualization.</p>}
                {!isLoading && imageUrls.length > 0 && (
                    <div className="flex flex-wrap gap-4 justify-center">
                        {imageUrls.map((url, i) => <img key={i} src={`${API_BASE_URL}${url}`} alt={`Visualization ${i+1}`} className="max-w-full md:max-w-md rounded-lg shadow-lg" />)}
                    </div>
                )}
            </div>
        </Section>
    );
};


const FinalSummary = ({ summaryData, classNames }) => {
    const { nomad_overall_metrics, role_model_performance } = summaryData;
    const MetricBox = ({ title, metrics }) => (<div className="bg-white p-6 rounded-lg shadow-md"><h4 className="text-xl font-bold text-gray-800 mb-4">{title}</h4><div className="space-y-2 text-gray-600"><p><strong>Accuracy:</strong> {metrics.accuracy?.toFixed(4) || 'N/A'}</p><p><strong>Avg F1-Score:</strong> {metrics.avg_metrics?.['f1-score']?.toFixed(4) || 'N/A'}</p><p><strong>Cost:</strong> {metrics.average_cost?.toFixed(4) || metrics.cost?.toFixed(2) || 'N/A'}</p></div></div>);
    const ConfusionMatrix = ({ title, cmData }) => {
        if (!cmData || cmData.length === 0 || !classNames || classNames.length === 0) return <p>Confusion Matrix not available.</p>;
        return (<div className="mt-4"><h5 className="font-semibold text-gray-700 mb-2">{title}</h5><div className="overflow-x-auto"><table className="w-full text-xs text-center border"><thead><tr><th className="p-2 border bg-gray-50">T\P</th>{classNames.map(l => <th key={l} title={l} className="p-2 border bg-gray-50 font-semibold truncate">{l}</th>)}</tr></thead><tbody>{cmData.map((row, i) => (<tr key={i}><td className="p-2 border bg-gray-50 font-semibold truncate" title={classNames[i]}>{classNames[i]}</td>{row.map((cell, j) => <td key={j} className="p-2 border">{cell}</td>)}</tr>))}</tbody></table></div></div>);
    };
    return (<Section title="Final Summary" step="6"><div className="grid grid-cols-1 md:grid-cols-2 gap-8"><MetricBox title="NOMAD Performance" metrics={nomad_overall_metrics} /><MetricBox title={`${role_model_performance.name} (Role Model)`} metrics={role_model_performance} /></div><div className="mt-8"><ConfusionMatrix title="NOMAD Confusion Matrix" cmData={nomad_overall_metrics.cm} /><ConfusionMatrix title="Role Model Confusion Matrix" cmData={role_model_performance.cm} /></div></Section>);
};

const FarmLVisualizer = () => {
    const [activeView, setActiveView] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [statusMessage, setStatusMessage] = useState('');
    const droneDataOutput = [ { drone: 0, data: [ { attack: 'Benign', train: 0, test: 60 }, { attack: 'Port Scan', train: 0, test: 68 }, { attack: 'ICMP Flood', train: 11, test: 0 }, { attack: 'Ping Sweep', train: 0, test: 72 }, { attack: 'Vulnerability Scan', train: 270, test: 0 }, { attack: 'OS Scan', train: 20, test: 0 }, { attack: 'DNS Flood', train: 0, test: 0 }, { attack: 'Slowloris', train: 13, test: 4 }, { attack: 'Dictionary Attack', train: 0, test: 419 }, { attack: 'UDP Flood', train: 0, test: 6 }, { attack: 'SYN Flood', train: 0, test: 0 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 314, test: 629 } }, { drone: 1, data: [ { attack: 'Benign', train: 0, test: 0 }, { attack: 'Port Scan', train: 0, test: 0 }, { attack: 'ICMP Flood', train: 0, test: 85 }, { attack: 'Ping Sweep', train: 5603, test: 1 }, { attack: 'Vulnerability Scan', train: 0, test: 0 }, { attack: 'OS Scan', train: 0, test: 0 }, { attack: 'DNS Flood', train: 5, test: 0 }, { attack: 'Slowloris', train: 7, test: 100 }, { attack: 'Dictionary Attack', train: 241, test: 34 }, { attack: 'UDP Flood', train: 0, test: 2 }, { attack: 'SYN Flood', train: 0, test: 6 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 5956, test: 228 } }, { drone: 2, data: [ { attack: 'Benign', train: 198, test: 0 }, { attack: 'Port Scan', train: 0, test: 0 }, { attack: 'ICMP Flood', train: 0, test: 0 }, { attack: 'Ping Sweep', train: 0, test: 0 }, { attack: 'Vulnerability Scan', train: 0, test: 0 }, { attack: 'OS Scan', train: 0, test: 3 }, { attack: 'DNS Flood', train: 3, test: 5415 }, { attack: 'Slowloris', train: 24056, test: 0 }, { attack: 'Dictionary Attack', train: 0, test: 0 }, { attack: 'UDP Flood', train: 0, test: 0 }, { attack: 'SYN Flood', train: 0, test: 0 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 24257, test: 5418 } }, { drone: 3, data: [ { attack: 'Benign', train: 5117, test: 0 }, { attack: 'Port Scan', train: 0, test: 0 }, { attack: 'ICMP Flood', train: 1, test: 0 }, { attack: 'Ping Sweep', train: 1373, test: 3 }, { attack: 'Vulnerability Scan', train: 36, test: 7 }, { attack: 'OS Scan', train: 1697, test: 1210 }, { attack: 'DNS Flood', train: 0, test: 5259 }, { attack: 'Slowloris', train: 0, test: 0 }, { attack: 'Dictionary Attack', train: 0, test: 0 }, { attack: 'UDP Flood', train: 0, test: 0 }, { attack: 'SYN Flood', train: 0, test: 0 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 8224, test: 6479 } }, { drone: 4, data: [ { attack: 'Benign', train: 6072, test: 0 }, { attack: 'Port Scan', train: 70, test: 0 }, { attack: 'ICMP Flood', train: 1, test: 2 }, { attack: 'Ping Sweep', train: 0, test: 8 }, { attack: 'Vulnerability Scan', train: 98, test: 693 }, { attack: 'OS Scan', train: 0, test: 522 }, { attack: 'DNS Flood', train: 0, test: 44 }, { attack: 'Slowloris', train: 8, test: 0 }, { attack: 'Dictionary Attack', train: 172, test: 0 }, { attack: 'UDP Flood', train: 0, test: 9 }, { attack: 'SYN Flood', train: 22, test: 0 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 6443, test: 1278 } }, { drone: 5, data: [ { attack: 'Benign', train: 1, test: 0 }, { attack: 'Port Scan', train: 271, test: 658 }, { attack: 'ICMP Flood', train: 334, test: 67 }, { attack: 'Ping Sweep', train: 0, test: 685 }, { attack: 'Vulnerability Scan', train: 0, test: 0 }, { attack: 'OS Scan', train: 164, test: 0 }, { attack: 'DNS Flood', train: 8, test: 31 }, { attack: 'Slowloris', train: 0, test: 224 }, { attack: 'Dictionary Attack', train: 4, test: 0 }, { attack: 'UDP Flood', train: 0, test: 0 }, { attack: 'SYN Flood', train: 0, test: 25 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 782, test: 1690 } }, { drone: 6, data: [ { attack: 'Benign', train: 23, test: 0 }, { attack: 'Port Scan', train: 2318, test: 141 }, { attack: 'ICMP Flood', train: 0, test: 0 }, { attack: 'Ping Sweep', train: 223, test: 3940 }, { attack: 'Vulnerability Scan', train: 0, test: 0 }, { attack: 'OS Scan', train: 1860, test: 0 }, { attack: 'DNS Flood', train: 0, test: 0 }, { attack: 'Slowloris', train: 0, test: 0 }, { attack: 'Dictionary Attack', train: 555, test: 0 }, { attack: 'UDP Flood', train: 0, test: 0 }, { attack: 'SYN Flood', train: 4, test: 0 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 4983, test: 4081 } }, { drone: 7, data: [ { attack: 'Benign', train: 1998, test: 0 }, { attack: 'Port Scan', train: 8, test: 0 }, { attack: 'ICMP Flood', train: 2, test: 0 }, { attack: 'Ping Sweep', train: 2162, test: 777 }, { attack: 'Vulnerability Scan', train: 22, test: 37 }, { attack: 'OS Scan', train: 0, test: 0 }, { attack: 'DNS Flood', train: 0, test: 0 }, { attack: 'Slowloris', train: 138, test: 8 }, { attack: 'Dictionary Attack', train: 82, test: 0 }, { attack: 'UDP Flood', train: 37, test: 0 }, { attack: 'SYN Flood', train: 1691, test: 932 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 6140, test: 1754 } }, { drone: 8, data: [ { attack: 'Benign', train: 5186, test: 0 }, { attack: 'Port Scan', train: 0, test: 277 }, { attack: 'ICMP Flood', train: 13, test: 1 }, { attack: 'Ping Sweep', train: 60, test: 0 }, { attack: 'Vulnerability Scan', train: 19, test: 177 }, { attack: 'OS Scan', train: 0, test: 17 }, { attack: 'DNS Flood', train: 858, test: 2 }, { attack: 'Slowloris', train: 621, test: 2 }, { attack: 'Dictionary Attack', train: 0, test: 0 }, { attack: 'UDP Flood', train: 0, test: 2 }, { attack: 'SYN Flood', train: 0, test: 0 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 6757, test: 478 } }, { drone: 9, data: [ { attack: 'Benign', train: 123, test: 7962 }, { attack: 'Port Scan', train: 1, test: 0 }, { attack: 'ICMP Flood', train: 1, test: 0 }, { attack: 'Ping Sweep', train: 3382, test: 0 }, { attack: 'Vulnerability Scan', train: 1688, test: 0 }, { attack: 'OS Scan', train: 345, test: 0 }, { attack: 'DNS Flood', train: 157, test: 0 }, { attack: 'Slowloris', train: 0, test: 0 }, { attack: 'Dictionary Attack', train: 6, test: 0 }, { attack: 'UDP Flood', train: 8, test: 0 }, { attack: 'SYN Flood', train: 529, test: 0 }, { attack: 'ARP Spoofing', train: 0, test: 0 } ], total: { train: 6240, test: 7962 } } ];

    const handleButtonClick = (viewName, loadingMessage, successMessage) => {
        const nextView = activeView === viewName ? null : viewName;
        setActiveView(nextView);

        if (nextView && loadingMessage) {
            setIsLoading(true);
            setStatusMessage(loadingMessage);
            setTimeout(() => {
                setStatusMessage(successMessage);
                setIsLoading(false);
            }, 1500);
        } else {
            setStatusMessage('');
        }
    };
    
    return (
        <Section title="FARML - Self-Supervised Federated Learning" step="7">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div>
                    <label className="font-semibold text-gray-700 block mb-1">Select Dataset</label>
                    <Select defaultValue="ACI IoT dataset 2023"><option>ACI IoT dataset 2023</option><option>UNSW dataset</option></Select>
                </div>
                 <div>
                    <label className="font-semibold text-gray-700 block mb-1">Number of Drones</label>
                    <Select defaultValue="10">{[...Array(10)].map((_, i) => (<option key={i+1}>{i+1}</option>))}</Select>
                </div>
                 <div>
                    <label className="font-semibold text-gray-700 block mb-1">Data Distribution</label>
                    <Select defaultValue="Low"><option>High</option><option>Medium</option><option>Low</option></Select>
                </div>
            </div>

            <div className="flex flex-wrap gap-4 items-center">
                <Button onClick={() => handleButtonClick('data', 'Preparing data...', 'Data preparation complete.')} disabled={isLoading} icon={ServerStackIcon}>{isLoading && activeView ==='data' ? 'Processing...' : 'Prepare Data'}</Button>
                <Button onClick={() => handleButtonClick('import', 'Importing model...', 'Model imported successfully.')} disabled={isLoading} icon={ArrowUpTrayIcon} className="bg-indigo-600 hover:bg-indigo-700">{isLoading && activeView === 'import' ? 'Importing...' : 'Import Model'}</Button>
                <Button onClick={() => handleButtonClick('architecture')} icon={CpuChipIcon} className="bg-gray-600 hover:bg-gray-700">{activeView === 'architecture' ? 'Hide' : 'Show'} Architecture</Button>
            </div>
            
            {statusMessage && <StatusMessage message={statusMessage} type="info" />}
            
            {activeView === 'data' && <DroneDataTable droneData={droneDataOutput} />}
            
            {activeView === 'import' && (
                <div className="mt-6">
                    <h4 className="text-lg font-semibold text-gray-700">Performance Metrics</h4>
                     <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
                        <div>
                          <img src="/federated-chart.png" alt="Federated Learning Comparison" className="w-full rounded-lg shadow-md" />
                           <p className="text-center text-sm text-gray-600 mt-2"><strong>Caption:</strong> ClusterFed converges efficiently with low communication overhead.</p>
                        </div>
                        <div>
                           <img src="/comm_cost.png" alt="Communication Cost" className="w-full rounded-lg shadow-md" />
                           <p className="text-center text-sm text-gray-600 mt-2"><strong>Caption:</strong> Communication cost analysis across different methods.</p>
                        </div>
                    </div>
                </div>
            )}
            {activeView === 'architecture' && (
                 <div className="mt-6">
                    <h4 className="text-lg font-semibold text-gray-700">System Architecture</h4>
                     <img src="/architecture-diagram.png" alt="System Architecture" className="w-full rounded-lg shadow-md mt-4" />
                </div>
            )}
        </Section>
    );
};

const DroneDataTable = ({ droneData }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const dataToShow = isExpanded ? droneData : droneData.slice(0, 3);
    
    return (
        <div className="mt-6">
            <div className="flex justify-between items-center mb-4">
                <h4 className="text-lg font-semibold text-gray-700">Data Distribution Across Drones</h4>
                <button onClick={() => setIsExpanded(!isExpanded)} className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1">
                    {isExpanded ? <ChevronUpIcon className="h-4 w-4" /> : <ChevronDownIcon className="h-4 w-4" />}
                    {isExpanded ? 'Collapse' : 'Expand All'}
                </button>
            </div>
            <div className="space-y-6">
            {dataToShow.map(drone => (
                <div key={drone.drone}>
                    <h5 className="font-bold text-gray-600 mb-2">Drone {drone.drone}</h5>
                    <div className="overflow-x-auto bg-white rounded-lg shadow">
                         <table className="w-full text-sm text-left text-gray-500">
                            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3">Attack Type</th>
                                    <th className="px-6 py-3">Train Samples</th>
                                    <th className="px-6 py-3">Test Samples</th>
                                </tr>
                            </thead>
                            <tbody>
                                {drone.data.map((item) => (
                                    <tr key={item.attack} className="bg-white border-b hover:bg-gray-50">
                                        <td className="px-6 py-4">{item.attack}</td>
                                        <td className="px-6 py-4">{item.train}</td>
                                        <td className="px-6 py-4">{item.test}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            ))}
            </div>
        </div>
    );
};


// --- Main App Component ---
export default function App() {
    const [models, setModels] = useState([]);
    const [status, setStatus] = useState({ message: '', type: 'info' });
    const [initialResults, setInitialResults] = useState(null);
    const [nomadConfigReady, setNomadConfigReady] = useState(false);
    const [simulationConfig, setSimulationConfig] = useState(null);
    const [simulationData, setSimulationData] = useState(null);
    const [finalSummary, setFinalSummary] = useState(null);
    
    const eventSourceRef = useRef(null);
    const chartDataRef = useRef({});

    const getDistinctColors = (count) => {
        const colors = [];
        for (let i = 0; i < count; i++) {
            const hue = (i * (360 / (count + 2))) % 360;
            colors.push(`hsl(${hue}, 85%, 50%)`);
        }
        return colors;
    };

    useEffect(() => {
        if (!simulationConfig) return;
        setSimulationData({ modelRunCounts: { labels: [], datasets: [] }, cumulativeCost: { labels: [], datasets: [] }, accuracyTrend: { labels: [], datasets: [] }, classPriors: { labels: [], datasets: [] } });
        setFinalSummary(null);
        setStatus({ message: 'Initializing simulation stream...', type: 'info' });
        const queryParams = new URLSearchParams(simulationConfig).toString();
        const es = new EventSource(`${API_BASE_URL}/api/nomad_event_stream?${queryParams}`);
        eventSourceRef.current = es;
        let roleModelStaticAccuracy = 0;
        let roleModelCost = 1;
        es.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'error') { setStatus({ message: `Stream Error: ${data.message}`, type: 'error' }); es.close(); return; }
            if (data.type === 'setup') {
                roleModelStaticAccuracy = data.role_model_static_accuracy;
                roleModelCost = data.role_model_cost_per_event || 1;
                const modelNames = data.candidate_model_names || [];
                const classNames = data.class_names_ordered || [];
                const classColors = getDistinctColors(classNames.length);

                chartDataRef.current = {
                    modelRunCounts: { labels: modelNames, datasets: [{ label: 'Times Executed', data: modelNames.map(() => 0), backgroundColor: 'rgba(59, 130, 246, 0.5)' }] },
                    cumulativeCost: { labels: [], datasets: [{ label: 'NOMAD Cost', data: [], borderColor: '#ef4444', tension: 0.1, backgroundColor: '#fecaca', fill: true }, { label: 'Role Model Cost', data: [], borderColor: '#3b82f6', tension: 0.1, borderDash: [5,5], backgroundColor: 'transparent' }] },
                    accuracyTrend: { labels: [], datasets: [{ label: 'NOMAD Accuracy', data: [], borderColor: '#22c55e', tension: 0.1, backgroundColor: '#dcfce7', fill: true }, { label: 'Role Model Accuracy', data: [], borderColor: '#f97316', tension: 0.1, borderDash: [5,5], backgroundColor: 'transparent' }] },
                    classPriors: { labels: [0], datasets: classNames.map((name, i) => ({ label: name, data: [data.initial_priors[name]], borderColor: classColors[i], tension: 0.1, fill: false })) }
                };
                setSimulationData(chartDataRef.current);
                setStatus({ message: `Simulation started. Total events: ${data.total_events}`, type: 'info' });

            } else if (data.type === 'batch_update') {
                const current = chartDataRef.current;
                const newLabels = [...current.cumulativeCost.labels, data.last_event_in_batch];
                current.modelRunCounts = { labels: Object.keys(data.model_run_counts_snapshot), datasets: [{ ...current.modelRunCounts.datasets[0], data: Object.values(data.model_run_counts_snapshot) }] };
                current.cumulativeCost = { ...current.cumulativeCost, labels: newLabels, datasets: [{...current.cumulativeCost.datasets[0], data: [...current.cumulativeCost.datasets[0].data, data.cumulative_cost]}, {...current.cumulativeCost.datasets[1], data: [...current.cumulativeCost.datasets[1].data, roleModelCost * data.last_event_in_batch]}] };
                current.accuracyTrend = { ...current.accuracyTrend, labels: newLabels, datasets: [{...current.accuracyTrend.datasets[0], data: [...current.accuracyTrend.datasets[0].data, data.live_nomad_accuracy]}, {...current.accuracyTrend.datasets[1], data: [...current.accuracyTrend.datasets[1].data, roleModelStaticAccuracy]}] };
                setSimulationData({ ...current });

            } else if (data.type === 'prior_update') {
                const current = chartDataRef.current;
                current.classPriors.labels.push(data.event_number);
                current.classPriors.datasets.forEach(dataset => {
                    dataset.data.push(data.updated_priors[dataset.label]);
                });
                setSimulationData({ ...current });

            } else if (data.type === 'summary') {
                setFinalSummary(data);
                setStatus({ message: 'Simulation complete!', type: 'success' });
                es.close();
            }
        };
        es.onerror = () => { setStatus({ message: 'Stream connection failed.', type: 'error' }); es.close(); };
        return () => { if (eventSourceRef.current) eventSourceRef.current.close(); };
    }, [simulationConfig]);

    return (
        <div className="bg-gray-100 min-h-screen font-sans">
            <header className="bg-white/80 backdrop-blur-lg sticky top-0 z-10 shadow-sm">
                <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 text-center">
                    <h1 className="text-4xl font-extrabold text-gray-800 tracking-tight flex items-center justify-center">
                        <img src="/nomad-logo.jpg" alt="NOMAD Logo" className="h-10 w-10 mr-3 rounded-full" />
                        NOMAD
                    </h1>
                    <p className="mt-2 text-lg text-gray-600 max-w-2xl mx-auto">An Interactive Visualizer for Adaptive Model Chaining</p>
                </div>
            </header>
            <main className="container mx-auto p-4 sm:p-6 lg:p-8">
                {status && status.message && <StatusMessage message={status.message} type={status.type} />}

                <ModelManagement models={models} setModels={setModels} setStatus={setStatus} />
                
                <DataUpload setStatus={setStatus} setInitialResults={setInitialResults} setNomadConfigReady={setNomadConfigReady} />
                
                {initialResults && (
                    <div className="pl-11">
                        <DataSummary initialResults={initialResults} />
                        <InitialTrainingSummary initialResults={initialResults}/>
                    </div>
                )}
                
                {initialResults && (
                    <CascadeVisualizer setStatus={setStatus}/>
                )}
                
                {nomadConfigReady && initialResults && (
                    <NomadConfig initialResults={initialResults} setSimulationConfig={setSimulationConfig} setStatus={setStatus}/>
                )}
                
                {simulationData && simulationData.modelRunCounts.labels.length > 0 && (
                    <Section title="Live Simulation Results" step="5">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            <ChartComponent type="bar" chartData={simulationData.modelRunCounts} chartOptions={{ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'Model Execution Frequency' }} }} />
                            <ChartComponent type="line" chartData={simulationData.cumulativeCost} chartOptions={{ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'Cumulative Cost' }}, scales: {y: {beginAtZero: true}} }} />
                            <ChartComponent type="line" chartData={simulationData.accuracyTrend} chartOptions={{ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'Accuracy Trend' }}, scales: {y: {min: 0, max: 1}} }} />
                            <ChartComponent type="line" chartData={simulationData.classPriors} chartOptions={{ responsive: true, maintainAspectRatio: false, plugins: { title: { display: true, text: 'Class Prior Distribution' }}, scales: {y: {min: 0, max: 1}} }} />
                        </div>
                    </Section>
                )}
                
                {finalSummary && (
                    <>
                        <FinalSummary summaryData={finalSummary} classNames={initialResults.classes_str} />
                        <FarmLVisualizer setStatus={setStatus} />
                    </>
                )}
                
            </main>
        </div>
    );
}
