import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import { ArrowUpTrayIcon, TrashIcon, PlusIcon, PlayIcon, InformationCircleIcon, ArrowPathIcon, Cog6ToothIcon, ChartBarIcon } from '@heroicons/react/24/outline';
import { CheckCircleIcon, XCircleIcon, ExclamationTriangleIcon } from '@heroicons/react/24/solid';

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
    ],
    LogisticRegression: [
      { name: 'C', type: 'number', placeholder: 'Regularization strength', defaultValue: 1.0 },
      { name: 'random_state', type: 'number', placeholder: 'Random State', defaultValue: 42 }
    ],
    GradientBoostingClassifier: [
      { name: 'n_estimators', type: 'number', placeholder: 'N Estimators (e.g., 100)', defaultValue: 100 },
      { name: 'learning_rate', type: 'number', placeholder: 'Learning Rate (e.g., 0.1)', defaultValue: 0.1 },
      { name: 'max_depth', type: 'number', placeholder: 'Max Depth (e.g., 3)', defaultValue: 3 }
    ]
};

// --- Helper & UI Components ---
const Section = ({ title, step, children, className = '' }) => (
    <section className={`bg-white/50 backdrop-blur-sm border border-gray-200/50 rounded-2xl shadow-lg p-6 md:p-8 mb-8 transition-all duration-300 ${className}`}>
        <h2 className="text-2xl font-bold text-gray-800 mb-2 flex items-center">
            <span className="bg-blue-600 text-white rounded-full h-8 w-8 flex items-center justify-center mr-3 text-lg">{step}</span>
            {title}
        </h2>
        <div className="pl-11">{children}</div>
    </section>
);

const Button = ({ children, onClick, disabled = false, icon, className = '', variant = 'primary' }) => {
    const baseClasses = "inline-flex items-center justify-center px-4 py-2 font-semibold rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-opacity-75 transition-all duration-200 disabled:cursor-not-allowed";
    const variants = {
        primary: "bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500 disabled:bg-gray-400",
        secondary: "bg-gray-600 text-white hover:bg-gray-700 focus:ring-gray-500 disabled:bg-gray-400",
        success: "bg-green-600 text-white hover:bg-green-700 focus:ring-green-500 disabled:bg-gray-400",
        warning: "bg-yellow-600 text-white hover:bg-yellow-700 focus:ring-yellow-500 disabled:bg-gray-400",
        danger: "bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 disabled:bg-gray-400"
    };
    
    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className={`${baseClasses} ${variants[variant]} ${className}`}
        >
            {icon && React.createElement(icon, { className: "h-5 w-5 mr-2" })}
            {children}
        </button>
    );
};

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
    const isWarning = type === 'warning';
    const Icon = isError ? XCircleIcon : (isWarning ? ExclamationTriangleIcon : (isInfo ? InformationCircleIcon : CheckCircleIcon));
    const colors = isError ? 'bg-red-100 border-red-400 text-red-700' : 
                   (isWarning ? 'bg-yellow-100 border-yellow-400 text-yellow-700' :
                   (isInfo ? 'bg-blue-100 border-blue-400 text-blue-700' : 'bg-green-100 border-green-400 text-green-700'));
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

// --- System Status Component ---
const SystemStatus = ({ setStatus }) => {
    const [systemStatus, setSystemStatus] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const fetchSystemStatus = useCallback(async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/api/system_status`);
            if (!response.ok) throw new Error('Failed to fetch system status');
            const data = await response.json();
            setSystemStatus(data);
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
        } finally {
            setIsLoading(false);
        }
    }, [setStatus]);

    useEffect(() => {
        fetchSystemStatus();
    }, [fetchSystemStatus]);

    if (!systemStatus && !isLoading) return null;

    return (
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
            <div className="flex justify-between items-center mb-3">
                <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                    <ChartBarIcon className="h-5 w-5 mr-2" />
                    System Status
                </h3>
                <Button onClick={fetchSystemStatus} disabled={isLoading} variant="secondary" className="text-xs">
                    <ArrowPathIcon className="h-4 w-4 mr-1" />
                    Refresh
                </Button>
            </div>
            {systemStatus && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className={`p-2 rounded ${systemStatus.system_ready ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        <div className="font-semibold">System Ready</div>
                        <div>{systemStatus.system_ready ? 'Yes' : 'No'}</div>
                    </div>
                    <div className="p-2 bg-blue-100 text-blue-800 rounded">
                        <div className="font-semibold">Models Loaded</div>
                        <div>{systemStatus.models_loaded}</div>
                    </div>
                    <div className={`p-2 rounded ${systemStatus.data_loaded ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                        <div className="font-semibold">Data Loaded</div>
                        <div>{systemStatus.data_loaded ? 'Yes' : 'No'}</div>
                    </div>
                    <div className={`p-2 rounded ${systemStatus.models_trained ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                        <div className="font-semibold">Models Trained</div>
                        <div>{systemStatus.models_trained ? systemStatus.trained_model_count : 'No'}</div>
                    </div>
                </div>
            )}
        </div>
    );
};

// --- Adaptive Configuration Component ---
const AdaptiveConfig = ({ setStatus, adaptiveConfig, setAdaptiveConfig }) => {
    const [config, setConfig] = useState({
        ph_threshold: 50.0,
        ph_delta: 0.005,
        buffer_size: 100,
        min_buffer_for_arima: 20,
        arima_order: [1, 0, 1],
        enable_arima: true
    });
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (adaptiveConfig) {
            setConfig(adaptiveConfig);
        } else {
            // Fetch default config
            fetchAdaptiveConfig();
        }
    }, [adaptiveConfig]);

    const fetchAdaptiveConfig = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/adaptive_config`);
            if (!response.ok) throw new Error('Failed to fetch adaptive config');
            const data = await response.json();
            setConfig(data);
            setAdaptiveConfig(data);
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
        }
    };

    const handleSaveConfig = async () => {
        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/api/adaptive_config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            if (!response.ok) throw new Error('Failed to save adaptive config');
            const data = await response.json();
            setAdaptiveConfig(config);
            setStatus({ message: data.message, type: 'success' });
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleTestAdaptive = async () => {
        setIsLoading(true);
        setStatus({ message: 'Testing adaptive functionality...', type: 'info' });
        try {
            const response = await fetch(`${API_BASE_URL}/api/test_adaptive`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ adaptive_config: config })
            });
            if (!response.ok) throw new Error('Adaptive test failed');
            const data = await response.json();
            setStatus({ 
                message: `Adaptive test completed. Events: ${data.test_results?.length || 0}, ARIMA: ${data.arima_available ? 'Available' : 'Not Available'}`, 
                type: 'success' 
            });
        } catch (error) {
            setStatus({ message: error.message, type: 'error' });
        } finally {
            setIsLoading(false);
        }
    };

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        if (name.startsWith('arima_order_')) {
            const index = parseInt(name.split('_')[2]);
            const newOrder = [...config.arima_order];
            newOrder[index] = parseInt(value) || 0;
            setConfig(prev => ({ ...prev, arima_order: newOrder }));
        } else if (type === 'checkbox') {
            setConfig(prev => ({ ...prev, [name]: checked }));
        } else if (type === 'number') {
            setConfig(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
        } else {
            setConfig(prev => ({ ...prev, [name]: value }));
        }
    };

    return (
        <div className="mt-8 pt-6 border-t">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-700 flex items-center">
                    <Cog6ToothIcon className="h-5 w-5 mr-2" />
                    Adaptive ARIMA Configuration
                </h3>
                <div className="space-x-2">
                    <Button onClick={handleTestAdaptive} disabled={isLoading} variant="warning">
                        Test Configuration
                    </Button>
                    <Button onClick={handleSaveConfig} disabled={isLoading}>
                        Save Configuration
                    </Button>
                </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Enable ARIMA
                    </label>
                    <div className="flex items-center">
                        <input
                            type="checkbox"
                            name="enable_arima"
                            checked={config.enable_arima}
                            onChange={handleInputChange}
                            className="h-4 w-4 text-blue-600 rounded"
                        />
                        <span className="ml-2 text-sm text-gray-600">Use ARIMA for drift detection</span>
                    </div>
                </div>
                
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Page-Hinkley Threshold
                    </label>
                    <Input
                        type="number"
                        name="ph_threshold"
                        value={config.ph_threshold}
                        onChange={handleInputChange}
                        step="1"
                        min="1"
                        placeholder="50.0"
                    />
                </div>
                
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Page-Hinkley Delta
                    </label>
                    <Input
                        type="number"
                        name="ph_delta"
                        value={config.ph_delta}
                        onChange={handleInputChange}
                        step="0.001"
                        min="0"
                        placeholder="0.005"
                    />
                </div>
                
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Buffer Size
                    </label>
                    <Input
                        type="number"
                        name="buffer_size"
                        value={config.buffer_size}
                        onChange={handleInputChange}
                        min="10"
                        placeholder="100"
                    />
                </div>
                
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        Min Buffer for ARIMA
                    </label>
                    <Input
                        type="number"
                        name="min_buffer_for_arima"
                        value={config.min_buffer_for_arima}
                        onChange={handleInputChange}
                        min="5"
                        placeholder="20"
                    />
                </div>
                
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                        ARIMA Order (p, d, q)
                    </label>
                    <div className="flex space-x-1">
                        {config.arima_order.map((value, index) => (
                            <Input
                                key={index}
                                type="number"
                                name={`arima_order_${index}`}
                                value={value}
                                onChange={handleInputChange}
                                min="0"
                                max="5"
                                className="w-full"
                                placeholder={`${['p', 'd', 'q'][index]}`}
                            />
                        ))}
                    </div>
                </div>
            </div>
            
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <h4 className="font-medium text-blue-900 mb-2">Configuration Guide:</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                    <li><strong>Page-Hinkley Threshold:</strong> Higher values make drift detection less sensitive</li>
                    <li><strong>Page-Hinkley Delta:</strong> Tolerance parameter for drift detection</li>
                    <li><strong>Buffer Size:</strong> Number of recent events to keep for ARIMA retraining</li>
                    <li><strong>ARIMA Order:</strong> (p, d, q) parameters where p=autoregressive, d=differencing, q=moving average</li>
                </ul>
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
                    className="hidden"
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


const NomadConfig = ({ initialResults, setSimulationConfig, setStatus, adaptiveConfig }) => {
    const [config, setConfig] = useState({ 
        role_model_name: '', 
        epsilon: 0.2, 
        quality_metric_for_ec: 'f1-score', 
        safety_check_type: 'relaxed',
        batch_size: 10,
        enable_dependent_models: false
    });
    const [phases, setPhases] = useState([]);

    useEffect(() => {
        if (initialResults.model_summaries?.length > 0) {
            const bestModel = [...initialResults.model_summaries].filter(m => !m.name.includes('FAILED')).sort((a, b) => b.accuracy - a.accuracy)[0];
            setConfig(c => ({...c, role_model_name: bestModel?.name || '' }));
        }
    }, [initialResults.model_summaries]);

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        if (type === 'checkbox') {
            setConfig(prev => ({ ...prev, [name]: checked }));
        } else if (type === 'number') {
            setConfig(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
        } else {
            setConfig(prev => ({ ...prev, [name]: value }));
        }
    };

    const handleRunSimulation = () => {
        if (!config.role_model_name) { 
            setStatus({ message: 'Please select a Role Model.', type: 'error' }); 
            return; 
        }
        
        // Validate phases
        for (const phase of phases) {
            const sum = Object.values(phase.target_distribution).reduce((a, b) => a + b, 0);
            if (Math.abs(sum - 1.0) > 0.01) { 
                setStatus({ message: `Probabilities in a phase must sum to 1.0.`, type: 'error' }); 
                return; 
            }
        }
        
        // Build simulation configuration with adaptive parameters
        const simConfig = {
            ...config,
            workload_phases: JSON.stringify(phases),
            // Add adaptive configuration parameters
            ph_threshold: adaptiveConfig?.ph_threshold || 50.0,
            ph_delta: adaptiveConfig?.ph_delta || 0.005,
            buffer_size: adaptiveConfig?.buffer_size || 100,
            min_buffer_for_arima: adaptiveConfig?.min_buffer_for_arima || 20,
            arima_order: adaptiveConfig?.arima_order ? adaptiveConfig.arima_order.join(',') : '1,0,1',
            enable_arima: adaptiveConfig?.enable_arima !== false ? 'true' : 'false'
        };
        
        setSimulationConfig(simConfig);
    };

    const addPhase = () => setPhases(p => [...p, { duration: 100, target_distribution: initialResults.initial_priors_str_keys || {} }]);
    const removePhase = (index) => setPhases(p => p.filter((_, i) => i !== index));
    const updatePhase = (index, field, value) => setPhases(p => p.map((phase, i) => i === index ? { ...phase, [field]: value } : phase));
    const handleResetPhaseDistribution = (phaseIndex) => updatePhase(phaseIndex, 'target_distribution', initialResults.initial_priors_str_keys);
    const updatePhaseDist = (phaseIndex, className, value) => { 
        setPhases(p => p.map((phase, i) => i === phaseIndex ? { 
            ...phase, 
            target_distribution: { ...phase.target_distribution, [className]: parseFloat(value) || 0 } 
        } : phase)); 
    };

    return (
        <Section title="Configure NOMAD Engine" step="3">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                <div>
                    <label className="font-semibold text-gray-700 block mb-1">Role Model</label>
                    <Select name="role_model_name" value={config.role_model_name} onChange={handleInputChange}>
                        <option value="">Select a Role Model</option>
                        {initialResults.model_summaries?.filter(m => !m.name.includes('FAILED')).map(m => (
                            <option key={m.name} value={m.name}>{`${m.name} (Acc: ${m.accuracy.toFixed(3)})`}</option>
                        ))}
                    </Select>
                </div>
                <div>
                    <label className="font-semibold text-gray-700 block mb-1">Epsilon (ε)</label>
                    <Input type="number" name="epsilon" value={config.epsilon} onChange={handleInputChange} step="0.01" min="0" />
                </div>
                <div>
                    <label className="font-semibold text-gray-700 block mb-1">Quality Metric</label>
                    <Select name="quality_metric_for_ec" value={config.quality_metric_for_ec} onChange={handleInputChange}>
                        <option value="f1-score">f1-score</option>
                        <option value="precision">precision</option>
                        <option value="recall">recall</option>
                    </Select>
                </div>
                <div>
                    <label className="font-semibold text-gray-700 block mb-1">Safety Check</label>
                    <Select name="safety_check_type" value={config.safety_check_type} onChange={handleInputChange}>
                        <option value="conservative">Conservative</option>
                        <option value="relaxed">Relaxed</option>
                    </Select>
                </div>
                <div>
                    <label className="font-semibold text-gray-700 block mb-1">Batch Size</label>
                    <Input type="number" name="batch_size" value={config.batch_size} onChange={handleInputChange} min="1" />
                </div>
                <div>
                    <label className="font-semibold text-gray-700 block mb-1">Enable Dependent Models</label>
                    <div className="flex items-center mt-2">
                        <input
                            type="checkbox"
                            name="enable_dependent_models"
                            checked={config.enable_dependent_models}
                            onChange={handleInputChange}
                            className="h-4 w-4 text-blue-600 rounded"
                        />
                        <span className="ml-2 text-sm text-gray-600">Support model dependencies</span>
                    </div>
                </div>
            </div>

            <div className="mt-8 pt-6 border-t">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Define Workload Phases (Optional)</h3>
                <div className="space-y-4">
                    {phases.map((phase, i) => (
                        <div key={i} className="p-4 bg-gray-50 rounded-lg border">
                            <div className="flex justify-between items-center mb-2">
                                <h4 className="font-bold text-gray-600">Phase {i+1}</h4>
                                <div className="flex items-center gap-3">
                                    <button onClick={() => handleResetPhaseDistribution(i)} className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1">
                                        <ArrowPathIcon className="h-4 w-4" /> Reset
                                    </button>
                                    <button onClick={() => removePhase(i)} className="text-red-500 hover:text-red-700">
                                        <TrashIcon className="h-5 w-5"/>
                                    </button>
                                </div>
                            </div>
                            <Input 
                                type="number" 
                                value={phase.duration} 
                                onChange={e => updatePhase(i, 'duration', parseInt(e.target.value))} 
                                placeholder="Duration (events)"
                                className="mb-3"
                            />
                            <div className="mt-2 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                                {Object.keys(phase.target_distribution).map(cn => (
                                    <div key={cn}>
                                        <label className="text-xs text-gray-500 truncate" title={cn}>{cn}</label>
                                        <Input 
                                            type="number" 
                                            value={phase.target_distribution[cn]} 
                                            onChange={e => updatePhaseDist(i, cn, e.target.value)} 
                                            step="0.01" 
                                            min="0" 
                                            max="1"
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
                <Button onClick={addPhase} icon={PlusIcon} variant="secondary" className="mt-4">
                    Add Phase
                </Button>
            </div>

            <div className="mt-8 text-center">
                <Button onClick={handleRunSimulation} icon={PlayIcon} variant="success" className="text-lg px-8 py-3">
                    Run NOMAD Simulation
                </Button>
            </div>
        </Section>
    );
};

const FinalSummary = ({ summaryData, classNames }) => {
    const { nomad_overall_metrics, role_model_performance, final_adaptive_stats } = summaryData;
    
    const MetricBox = ({ title, metrics }) => (
        <div className="bg-white p-6 rounded-lg shadow-md">
            <h4 className="text-xl font-bold text-gray-800 mb-4">{title}</h4>
            <div className="space-y-2 text-gray-600">
                <p><strong>Accuracy:</strong> {metrics.accuracy?.toFixed(4) || 'N/A'}</p>
                <p><strong>Avg F1-Score:</strong> {metrics.avg_metrics?.['f1-score']?.toFixed(4) || 'N/A'}</p>
                <p><strong>Cost:</strong> {metrics.average_cost?.toFixed(4) || metrics.cost?.toFixed(2) || 'N/A'}</p>
            </div>
        </div>
    );
    
    const AdaptiveStatsBox = ({ stats }) => (
        <div className="bg-white p-6 rounded-lg shadow-md">
            <h4 className="text-xl font-bold text-gray-800 mb-4">Adaptive Statistics</h4>
            <div className="space-y-2 text-gray-600">
                <p><strong>Total Events:</strong> {stats?.total_events || 'N/A'}</p>
                <p><strong>Drift Detections:</strong> {stats?.drift_count || 'N/A'}</p>
                <p><strong>ARIMA Enabled:</strong> {stats?.arima_enabled ? 'Yes' : 'No'}</p>
                <p><strong>ARIMA Models:</strong> {stats?.arima_models_count || 'N/A'}</p>
                <p><strong>Buffer Size:</strong> {stats?.buffer_size || 'N/A'}</p>
            </div>
        </div>
    );
    
    const ConfusionMatrix = ({ title, cmData }) => {
        if (!cmData || cmData.length === 0 || !classNames || classNames.length === 0) return <p>Confusion Matrix not available.</p>;
        return (
            <div className="mt-4">
                <h5 className="font-semibold text-gray-700 mb-2">{title}</h5>
                <div className="overflow-x-auto">
                    <table className="w-full text-xs text-center border">
                        <thead>
                            <tr>
                                <th className="p-2 border bg-gray-50">T\P</th>
                                {classNames.map(l => <th key={l} title={l} className="p-2 border bg-gray-50 font-semibold truncate">{l}</th>)}
                            </tr>
                        </thead>
                        <tbody>
                            {cmData.map((row, i) => (
                                <tr key={i}>
                                    <td className="p-2 border bg-gray-50 font-semibold truncate" title={classNames[i]}>{classNames[i]}</td>
                                    {row.map((cell, j) => <td key={j} className="p-2 border">{cell}</td>)}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        );
    };
    
    return (
        <Section title="Final Summary" step="5">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
                <MetricBox title="NOMAD Performance" metrics={nomad_overall_metrics} />
                <MetricBox title={`${role_model_performance.name} (Role Model)`} metrics={role_model_performance} />
                {final_adaptive_stats && <AdaptiveStatsBox stats={final_adaptive_stats} />}
            </div>
            <div className="mt-8">
                <ConfusionMatrix title="NOMAD Confusion Matrix" cmData={nomad_overall_metrics.cm} />
                <ConfusionMatrix title="Role Model Confusion Matrix" cmData={role_model_performance.cm} />
            </div>
        </Section>
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
    const [adaptiveConfig, setAdaptiveConfig] = useState(null);
    const [driftEvents, setDriftEvents] = useState([]);
    
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
        
        setSimulationData({ 
            modelRunCounts: { labels: [], datasets: [] }, 
            cumulativeCost: { labels: [], datasets: [] }, 
            accuracyTrend: { labels: [], datasets: [] }, 
            classPriors: { labels: [], datasets: [] } 
        });
        setFinalSummary(null);
        setDriftEvents([]);
        setStatus({ message: 'Initializing simulation stream...', type: 'info' });
        
        const queryParams = new URLSearchParams(simulationConfig).toString();
        const es = new EventSource(`${API_BASE_URL}/api/nomad_event_stream?${queryParams}`);
        eventSourceRef.current = es;
        let roleModelStaticAccuracy = 0;
        let roleModelCost = 1;

        es.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'error') { 
                setStatus({ message: `Stream Error: ${data.message}`, type: 'error' }); 
                es.close(); 
                return; 
            }
            
            if (data.type === 'setup') {
                roleModelStaticAccuracy = data.role_model_static_accuracy;
                roleModelCost = data.role_model_cost_per_event || 1;
                const modelNames = data.candidate_model_names || [];
                const classNames = data.class_names_ordered || [];
                const classColors = getDistinctColors(classNames.length);

                chartDataRef.current = {
                    modelRunCounts: { 
                        labels: modelNames, 
                        datasets: [{ 
                            label: 'Times Executed', 
                            data: modelNames.map(() => 0), 
                            backgroundColor: 'rgba(59, 130, 246, 0.5)' 
                        }] 
                    },
                    cumulativeCost: { 
                        labels: [], 
                        datasets: [
                            { 
                                label: 'NOMAD Cost', 
                                data: [], 
                                borderColor: '#ef4444', 
                                tension: 0.1, 
                                backgroundColor: '#fecaca', 
                                fill: true 
                            }, 
                            { 
                                label: 'Role Model Cost', 
                                data: [], 
                                borderColor: '#3b82f6', 
                                tension: 0.1, 
                                borderDash: [5,5], 
                                backgroundColor: 'transparent' 
                            }
                        ] 
                    },
                    accuracyTrend: { 
                        labels: [], 
                        datasets: [
                            { 
                                label: 'NOMAD Accuracy', 
                                data: [], 
                                borderColor: '#22c55e', 
                                tension: 0.1, 
                                backgroundColor: '#dcfce7', 
                                fill: true 
                            }, 
                            { 
                                label: 'Role Model Accuracy', 
                                data: [], 
                                borderColor: '#f97316', 
                                tension: 0.1, 
                                borderDash: [5,5], 
                                backgroundColor: 'transparent' 
                            }
                        ] 
                    },
                    classPriors: { 
                        labels: [0], 
                        datasets: classNames.map((name, i) => ({ 
                            label: name, 
                            data: [data.initial_priors[name]], 
                            borderColor: classColors[i], 
                            tension: 0.1, 
                            fill: false 
                        })) 
                    }
                };
                setSimulationData(chartDataRef.current);
                setStatus({ message: `Simulation started. Total events: ${data.total_events}`, type: 'info' });

            } else if (data.type === 'batch_update') {
                const current = chartDataRef.current;
                const newLabels = [...current.cumulativeCost.labels, data.last_event_in_batch];
                
                current.modelRunCounts = { 
                    labels: Object.keys(data.model_run_counts_snapshot), 
                    datasets: [{ 
                        ...current.modelRunCounts.datasets[0], 
                        data: Object.values(data.model_run_counts_snapshot) 
                    }] 
                };
                
                current.cumulativeCost = { 
                    ...current.cumulativeCost, 
                    labels: newLabels, 
                    datasets: [
                        {...current.cumulativeCost.datasets[0], data: [...current.cumulativeCost.datasets[0].data, data.cumulative_cost]}, 
                        {...current.cumulativeCost.datasets[1], data: [...current.cumulativeCost.datasets[1].data, roleModelCost * data.last_event_in_batch]}
                    ] 
                };
                
                current.accuracyTrend = { 
                    ...current.accuracyTrend, 
                    labels: newLabels, 
                    datasets: [
                        {...current.accuracyTrend.datasets[0], data: [...current.accuracyTrend.datasets[0].data, data.live_nomad_accuracy]}, 
                        {...current.accuracyTrend.datasets[1], data: [...current.accuracyTrend.datasets[1].data, roleModelStaticAccuracy]}
                    ] 
                };

                // Update class priors
                if (data.current_priors) {
                    current.classPriors.labels.push(data.last_event_in_batch);
                    current.classPriors.datasets.forEach(dataset => {
                        dataset.data.push(data.current_priors[dataset.label] || 0);
                    });
                }
                
                setSimulationData({ ...current });

            } else if (data.type === 'drift_detected') {
                setDriftEvents(prev => [...prev, {
                    event_number: data.event_number,
                    updated_priors: data.updated_priors,
                    adaptive_stats: data.adaptive_stats,
                    timestamp: new Date().toLocaleTimeString()
                }]);
                setStatus({ 
                    message: `Drift detected at event ${data.event_number}! Priors updated.`, 
                    type: 'warning' 
                });

            } else if (data.type === 'phase_shift') {
                setStatus({ 
                    message: `Phase ${data.phase_index + 1} started (${data.duration} events, KL divergence: ${data.kl_divergence_from_previous?.toFixed(3) || 'N/A'})`, 
                    type: 'info' 
                });

            } else if (data.type === 'summary') {
                setFinalSummary(data);
                setStatus({ message: 'Simulation complete!', type: 'success' });
                es.close();
            }
        };
        
        es.onerror = () => { 
            setStatus({ message: 'Stream connection failed.', type: 'error' }); 
            es.close(); 
        };
        
        return () => { 
            if (eventSourceRef.current) eventSourceRef.current.close(); 
        };
    }, [simulationConfig]);

    return (
        <div className="bg-gray-100 min-h-screen font-sans">
            <header className="bg-white/80 backdrop-blur-lg sticky top-0 z-10 shadow-sm">
                <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 text-center">
                    <h1 className="text-4xl font-extrabold text-gray-800 tracking-tight flex items-center justify-center">
                        <span className="bg-gradient-to-r from-blue-600 to-purple-600 text-transparent bg-clip-text mr-3">
                            NOMAD
                        </span>
                        <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded-full">with ARIMA</span>
                    </h1>
                    <p className="mt-2 text-lg text-gray-600 max-w-2xl mx-auto">
                        An Interactive Visualizer for Adaptive Model Chaining with Drift Detection
                    </p>
                </div>
            </header>

            <main className="container mx-auto p-4 sm:p-6 lg:p-8">
                {status && status.message && <StatusMessage message={status.message} type={status.type} />}

                <SystemStatus setStatus={setStatus} />

                <ModelManagement models={models} setModels={setModels} setStatus={setStatus} />
                
                <DataUpload setStatus={setStatus} setInitialResults={setInitialResults} setNomadConfigReady={setNomadConfigReady} />
                
                {initialResults && (
                    <div className="pl-11">
                        <DataSummary initialResults={initialResults} />
                        <InitialTrainingSummary initialResults={initialResults}/>
                        <AdaptiveConfig 
                            setStatus={setStatus} 
                            adaptiveConfig={adaptiveConfig} 
                            setAdaptiveConfig={setAdaptiveConfig} 
                        />
                    </div>
                )}
                
                {nomadConfigReady && initialResults && (
                    <NomadConfig 
                        initialResults={initialResults} 
                        setSimulationConfig={setSimulationConfig} 
                        setStatus={setStatus}
                        adaptiveConfig={adaptiveConfig}
                    />
                )}

                {driftEvents.length > 0 && (
                    <Section title="Drift Detection Events" step="4a" className="border-l-4 border-yellow-500">
                        <div className="space-y-3">
                            {driftEvents.slice(-5).map((event, index) => (
                                <div key={index} className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                                    <div className="flex justify-between items-start mb-2">
                                        <span className="font-semibold text-yellow-800">
                                            Event #{event.event_number}
                                        </span>
                                        <span className="text-sm text-yellow-600">{event.timestamp}</span>
                                    </div>
                                    <p className="text-sm text-yellow-700 mb-2">
                                        Drift detected - ARIMA models retrained
                                    </p>
                                    {event.adaptive_stats && (
                                        <div className="text-xs text-yellow-600 grid grid-cols-2 gap-2">
                                            <span>Total Drifts: {event.adaptive_stats.drift_count}</span>
                                            <span>Buffer Size: {event.adaptive_stats.buffer_size}</span>
                                        </div>
                                    )}
                                </div>
                            ))}
                            {driftEvents.length > 5 && (
                                <p className="text-sm text-gray-500 text-center">
                                    Showing last 5 of {driftEvents.length} drift events
                                </p>
                            )}
                        </div>
                    </Section>
                )}
                
                {simulationData && simulationData.modelRunCounts.labels.length > 0 && (
                    <Section title="Live Simulation Results" step="4">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                            <ChartComponent 
                                type="bar" 
                                chartData={simulationData.modelRunCounts} 
                                chartOptions={{ 
                                    responsive: true, 
                                    maintainAspectRatio: false, 
                                    plugins: { title: { display: true, text: 'Model Execution Frequency' }} 
                                }} 
                            />
                            <ChartComponent 
                                type="line" 
                                chartData={simulationData.cumulativeCost} 
                                chartOptions={{ 
                                    responsive: true, 
                                    maintainAspectRatio: false, 
                                    plugins: { title: { display: true, text: 'Cumulative Cost' }}, 
                                    scales: {y: {beginAtZero: true}} 
                                }} 
                            />
                            <ChartComponent 
                                type="line" 
                                chartData={simulationData.accuracyTrend} 
                                chartOptions={{ 
                                    responsive: true, 
                                    maintainAspectRatio: false, 
                                    plugins: { title: { display: true, text: 'Accuracy Trend' }}, 
                                    scales: {y: {min: 0, max: 1}} 
                                }} 
                            />
                            <ChartComponent 
                                type="line" 
                                chartData={simulationData.classPriors} 
                                chartOptions={{ 
                                    responsive: true, 
                                    maintainAspectRatio: false, 
                                    plugins: { title: { display: true, text: 'Adaptive Class Priors' }}, 
                                    scales: {y: {min: 0, max: 1}} 
                                }} 
                            />
                        </div>
                    </Section>
                )}
                
                {finalSummary && (
                    <FinalSummary summaryData={finalSummary} classNames={initialResults.classes_str} />
                )}
                
            </main>
        </div>
    );
}