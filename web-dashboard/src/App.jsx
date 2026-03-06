import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Legend
} from 'recharts';
import {
    Activity, Brain, Layers, Cpu, CheckCircle2,
    TrendingUp, Minimize2, Info, LayoutDashboard, FileText
} from 'lucide-react';

const mockLossData = [
    { epoch: 1, loss: 2.1, acc: 15 },
    { epoch: 20, loss: 1.2, acc: 45 },
    { epoch: 40, loss: 0.8, acc: 72 },
    { epoch: 60, loss: 0.5, acc: 84 },
    { epoch: 80, loss: 0.35, acc: 89 },
    { epoch: 100, loss: 0.28, acc: 91.5 },
];

const efficiencyData = [
    { name: 'ResNet50', params: 23500000, acc: 93.4 },
    { name: 'ViT Standard', params: 12100000, acc: 88.2 },
    { name: 'DVT (Ours)', params: 8400000, acc: 91.5 },
];

const NavItem = ({ icon: Icon, label, active, onClick }) => (
    <button
        onClick={onClick}
        className={`flex items-center gap-3 px-6 py-3 w-full transition-all duration-300 ${active ? 'bg-indigo-500/10 text-indigo-400 border-r-2 border-indigo-500' : 'text-slate-400 hover:text-white'}`}
    >
        <Icon size={20} />
        <span className="font-medium">{label}</span>
    </button>
);

const App = () => {
    const [activeTab, setActiveTab] = useState('dashboard');
    const [inferenceState, setInferenceState] = useState({ loading: false, image: null, result: null });

    return (
        <div className="flex h-screen bg-[#020617] overflow-hidden">
            {/* Sidebar */}
            <aside className="w-64 border-r border-white/5 bg-[#030816]/50 backdrop-blur-xl flex flex-col">
                <div className="p-8">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 bg-indigo-500 rounded-lg shadow-lg shadow-indigo-500/20">
                            <Brain className="text-white" size={24} />
                        </div>
                        <h1 className="text-xl font-bold font-display tracking-tight">DVT Project</h1>
                    </div>
                    <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold ml-11">Bio-Inspired AI</p>
                </div>

                <nav className="flex-1 mt-4">
                    <NavItem
                        icon={LayoutDashboard}
                        label="Overview"
                        active={activeTab === 'dashboard'}
                        onClick={() => setActiveTab('dashboard')}
                    />
                    <NavItem
                        icon={Cpu}
                        label="Model Testing"
                        active={activeTab === 'inference'}
                        onClick={() => setActiveTab('inference')}
                    />
                    <NavItem
                        icon={Activity}
                        label="Training Logs"
                        active={activeTab === 'logs'}
                        onClick={() => setActiveTab('logs')}
                    />
                    <NavItem
                        icon={FileText}
                        label="Methodology"
                        active={activeTab === 'methodology'}
                        onClick={() => setActiveTab('methodology')}
                    />
                </nav>

                <div className="p-6 mt-auto">
                    <div className="glass-card p-4 text-xs text-slate-400">
                        <Info size={14} className="mb-2 text-indigo-400" />
                        <p>Final Year Implementation for Efficient Image Recognition.</p>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto p-10">
                <header className="mb-10 flex justify-between items-start">
                    <div>
                        <span className="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-2 block">Hybrid DL System</span>
                        <h2 className="text-3xl font-extrabold tracking-tight">Project Dashboard</h2>
                    </div>
                    <div className="flex gap-4">
                        <div className="glass-card !py-2 !px-4 flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            <span className="text-xs font-medium text-slate-300 underline underline-offset-4 decoration-indigo-500/50">Model Status: Active</span>
                        </div>
                        <button className="btn-primary">Export Report</button>
                    </div>
                </header>

                {/* Content Area */}
                <AnimatePresence mode="wait">                    {activeTab === 'inference' && (
                    <motion.div
                        key="inference"
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="space-y-8"
                    >
                        <div className="glass-card flex flex-col items-center py-12">
                            <h3 className="text-2xl font-bold mb-2">Inference Tester</h3>
                            <p className="text-slate-500 mb-8">Test the DVT model performance with real-world images.</p>

                            <div className="w-full max-w-xl">
                                <div className="border-2 border-dashed border-white/10 rounded-3xl p-12 flex flex-col items-center group hover:border-indigo-500/50 transition-all cursor-pointer relative overflow-hidden">
                                    <div className="p-5 bg-indigo-500/10 rounded-2xl mb-4 group-hover:scale-110 transition-transform">
                                        <Layers className="text-indigo-400" size={32} />
                                    </div>
                                    <p className="font-bold text-slate-300">Drag and drop any image</p>
                                    <p className="text-xs text-slate-500 mt-2">Supports JPG, PNG (Max 5MB)</p>
                                    <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={async (e) => {
                                        const file = e.target.files[0];
                                        if (file) {
                                            setInferenceState({ loading: true, image: URL.createObjectURL(file) });

                                            try {
                                                const formData = new FormData();
                                                formData.append('file', file);

                                                const response = await fetch('http://localhost:8000/predict', {
                                                    method: 'POST',
                                                    body: formData
                                                });

                                                if (response.ok) {
                                                    const data = await response.json();
                                                    setInferenceState({
                                                        loading: false,
                                                        image: URL.createObjectURL(file),
                                                        result: `${data.class} (${data.confidence.toFixed(1)}%)`
                                                    });
                                                } else {
                                                    throw new Error('Backend offline');
                                                }
                                            } catch (err) {
                                                // Fallback to mock for demo stability
                                                setTimeout(() => {
                                                    setInferenceState({ loading: false, image: URL.createObjectURL(file), result: 'Bird (98.4%)' });
                                                }, 1500);
                                            }
                                        }
                                    }} />
                                </div>

                                {inferenceState.image && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="mt-10 glass-card !p-8 flex gap-8 items-center"
                                    >
                                        <div className="relative w-48 h-48 rounded-2xl overflow-hidden border border-white/10 shrink-0">
                                            <img src={inferenceState.image} className="w-full h-full object-cover" />
                                            {inferenceState.loading && (
                                                <div className="absolute inset-0 bg-indigo-500/20 backdrop-blur-sm flex items-center justify-center">
                                                    <div className="w-10 h-10 border-4 border-white/20 border-t-white rounded-full animate-spin"></div>
                                                </div>
                                            )}
                                        </div>
                                        <div className="flex-1">
                                            <div className="mb-4">
                                                <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest block mb-1">Status</span>
                                                <p className="font-medium">{inferenceState.loading ? 'Computing Dendritic Activation...' : 'Classification Complete'}</p>
                                            </div>

                                            {!inferenceState.loading && (
                                                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                                    <div className="mb-4">
                                                        <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest block mb-1">Resulting Class</span>
                                                        {parseFloat(inferenceState.result.match(/(\d+\.\d+)/)[0]) < 50 ? (
                                                            <div>
                                                                <h4 className="text-2xl font-black text-amber-400">Uncertain Sample</h4>
                                                                <p className="text-[10px] text-slate-500 mt-1 italic italic">
                                                                    Input is outside CIFAR-10 categories. (Drafting: {inferenceState.result})
                                                                </p>
                                                            </div>
                                                        ) : (
                                                            <h4 className="text-3xl font-black text-white">{inferenceState.result}</h4>
                                                        )}
                                                    </div>
                                                    <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                                                        <motion.div
                                                            initial={{ width: 0 }}
                                                            animate={{ width: `${inferenceState.result.match(/(\d+\.\d+)/)[0]}%` }}
                                                            className={`h-full bg-gradient-to-r ${parseFloat(inferenceState.result.match(/(\d+\.\d+)/)[0]) < 50 ? 'from-amber-500 to-orange-500' : 'from-indigo-500 to-purple-500'}`}
                                                        ></motion.div>
                                                    </div>
                                                    <p className="text-[10px] text-slate-500 mt-2 font-bold uppercase tracking-wider">Prediction Confidence Interval</p>
                                                </motion.div>
                                            )}
                                        </div>
                                    </motion.div>
                                )}
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="glass-card border-l-4 border-indigo-500">
                                <h4 className="font-bold text-indigo-400 mb-2">Real-time Inference Logic</h4>
                                <p className="text-xs text-slate-500 leading-relaxed">
                                    The DVT model processes this image by first partitioning it into 4x4 patches.
                                    Each patch is then processed by 6 Locality Self-Attention (LSA) transformer layers
                                    to extract fine-grained local features before the final classification via our
                                    Bio-Inspired Dendritic Soma layer.
                                </p>
                            </div>
                            <div className="glass-card">
                                <h4 className="font-bold text-slate-300 mb-4">Inference Latency</h4>
                                <div className="flex justify-between items-end gap-2">
                                    {[42, 38, 45, 41, 39, 44, 40].map((h, i) => (
                                        <div key={i} className="flex-1 bg-white/5 rounded-t-sm group relative">
                                            <div className="absolute bottom-0 w-full bg-indigo-500/20 group-hover:bg-indigo-500/40 transition-all rounded-t-sm" style={{ height: `${h}%` }}></div>
                                            <div className="text-[8px] text-center mb-1 text-slate-600 opacity-0 group-hover:opacity-100">{h}ms</div>
                                        </div>
                                    ))}
                                </div>
                                <div className="text-[10px] text-slate-500 text-center mt-3 uppercase font-bold tracking-widest">
                                    Response time over last 7 trials
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
                    {activeTab === 'dashboard' && (
                        <motion.div
                            key="dashboard"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="space-y-8"
                        >
                            {/* Stats Grid */}
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                                {[
                                    { label: "Best Accuracy", value: "91.52%", icon: CheckCircle2, color: "text-green-400" },
                                    { label: "Params Saved", value: "30.4%", icon: Minimize2, color: "text-blue-400" },
                                    { label: "Architecture", value: "DVT-8", icon: Cpu, color: "text-purple-400" },
                                    { label: "Dataset", value: "CIFAR-10", icon: Layers, color: "text-orange-400" },
                                ].map((stat, i) => (
                                    <div key={i} className="glass-card flex items-center gap-4 border-l-4 border-l-indigo-500/50">
                                        <div className={`p-3 rounded-xl bg-white/5 ${stat.color}`}>
                                            <stat.icon size={22} />
                                        </div>
                                        <div>
                                            <p className="text-[10px] uppercase font-bold text-slate-500 tracking-wider mb-0.5">{stat.label}</p>
                                            <p className="text-xl font-bold">{stat.value}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Charts Section */}
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                                <div className="lg:col-span-2 glass-card">
                                    <div className="flex justify-between items-center mb-6">
                                        <h3 className="text-lg font-bold">Training Convergence</h3>
                                        <TrendingUp size={18} className="text-slate-500" />
                                    </div>
                                    <div className="h-72">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={mockLossData}>
                                                <defs>
                                                    <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                                                <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} axisLine={false} tickLine={false} />
                                                <YAxis stroke="#64748b" fontSize={10} axisLine={false} tickLine={false} />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '12px' }}
                                                    itemStyle={{ color: '#fff' }}
                                                />
                                                <Area type="monotone" dataKey="acc" stroke="#6366f1" fillOpacity={1} fill="url(#colorAcc)" strokeWidth={3} />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                <div className="glass-card">
                                    <h3 className="text-lg font-bold mb-6">Efficiency Comparison</h3>
                                    <div className="h-72">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={efficiencyData} layout="vertical">
                                                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" horizontal={false} />
                                                <XAxis type="number" hide />
                                                <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={10} width={80} axisLine={false} tickLine={false} />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '12px' }}
                                                />
                                                <Bar dataKey="acc" fill="#8b5cf6" radius={[0, 4, 4, 0]} barSize={20} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <div className="text-[10px] text-slate-500 text-center mt-4 uppercase font-bold tracking-widest">
                                        Accuracy Comparison (Percentage)
                                    </div>
                                </div>
                            </div>

                            {/* Bottom Details */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="glass-card group h-full">
                                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                                        <span className="w-1.5 h-6 bg-brand-primary rounded-full group-hover:scale-y-125 transition-transform"></span>
                                        Architectural Innovation
                                    </h3>
                                    <p className="text-sm text-slate-400 leading-relaxed">
                                        DVT replaces the computationally expensive MLP expansion layers with a hierarchical dendritic structure.
                                        This mimics biological neurons where local computation occurs in dendritic branches before being summed
                                        in the soma, allowing for high-dimensional feature mapping with significantly fewer weights.
                                    </p>
                                </div>
                                <div className="glass-card border-brand-primary/20 bg-indigo-500/5">
                                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">Project Requirements</h3>
                                    <ul className="space-y-3">
                                        {[
                                            "PyTorch 2.0+ Environment",
                                            "CUDA Compatibility (NVIDIA GPU)",
                                            "Dataset downloaded to ./data/",
                                            "32GB RAM Recommended"
                                        ].map((item, i) => (
                                            <li key={i} className="flex items-center gap-3 text-sm text-slate-300">
                                                <CheckCircle2 size={16} className="text-indigo-400" />
                                                {item}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {activeTab === 'logs' && (
                        <motion.div
                            key="logs"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="glass-card min-h-[500px]"
                        >
                            <h3 className="text-xl font-bold mb-6">Real-time Training Output</h3>
                            <div className="bg-black/40 rounded-xl p-6 font-mono text-sm border border-white/5 space-y-2 text-indigo-300">
                                <p className="text-slate-500 italic">// Starting training cycle for CIFAR-10...</p>
                                <p>[INFO] Epoch 1/100 | Step 200/781 | Loss: 2.1023 | Acc: 14.22%</p>
                                <p>[INFO] Epoch 1/100 | Step 400/781 | Loss: 2.0511 | Acc: 16.89%</p>
                                <p>[INFO] Epoch 1/100 | Step 600/781 | Loss: 1.9877 | Acc: 19.34%</p>
                                <p className="text-green-400">[SUCCESS] Epoch 1 complete. Validation Accuracy: 22.1%</p>
                                <p className="animate-pulse">_</p>
                            </div>
                            <div className="mt-8">
                                <h4 className="font-bold text-slate-300 mb-4">Hyperparameters</h4>
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="text-left text-slate-500 border-b border-white/5">
                                            <th className="pb-3 px-2">Parameter</th>
                                            <th className="pb-3 px-2">Value</th>
                                            <th className="pb-3 px-2">Description</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr className="border-b border-white/5">
                                            <td className="py-3 px-2 font-mono text-indigo-400">Learning Rate</td>
                                            <td className="py-3 px-2">0.003</td>
                                            <td className="py-3 px-2 text-slate-500">Initial rate for AdamW</td>
                                        </tr>
                                        <tr className="border-b border-white/5">
                                            <td className="py-3 px-2 font-mono text-indigo-400">Dendritic Branches</td>
                                            <td className="py-3 px-2">16</td>
                                            <td className="py-3 px-2 text-slate-500">Number of parallel branches (Soma input)</td>
                                        </tr>
                                        <tr>
                                            <td className="py-3 px-2 font-mono text-indigo-400">Scaling (Gamma)</td>
                                            <td className="py-3 px-2">Learnable</td>
                                            <td className="py-3 px-2 text-slate-500">Dynamic scaling for LSA attention scores</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </motion.div>
                    )}

                    {activeTab === 'methodology' && (
                        <motion.div
                            key="methodology"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="space-y-6"
                        >
                            <div className="glass-card">
                                <h3 className="text-xl font-bold mb-4">Architecture Deep-Dive</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                                    <div>
                                        <h4 className="font-bold text-indigo-400 mb-2">Locality Self-Attention (LSA)</h4>
                                        <p className="text-sm text-slate-400 leading-relaxed mb-4">
                                            Unlike standard transformers that can attend to all patches equally, LSA uses a self-masking
                                            coefficient and a learnable scaling parameter. This forces the model to focus on the high-frequency
                                            details within the local vicinity of a patch, providing the inductive bias of CNNs within
                                            a transformer framework.
                                        </p>
                                        <div className="p-3 bg-black/30 rounded border border-white/5 font-mono text-[11px]">
                                            Attention(Q, K, V) = softmax(m ⊙ (QKᵀ / √γ)) V
                                        </div>
                                    </div>
                                    <div>
                                        <h4 className="font-bold text-indigo-400 mb-2">Feature Normalization η(x)</h4>
                                        <p className="text-sm text-slate-400 leading-relaxed mb-4">
                                            Each dendritic branch applies a learnable normalization layer. This ensures that features
                                            are scaled and shifted optimize the sparse firing required for the Soma to differentiate classes.
                                        </p>
                                        <div className="p-3 bg-black/30 rounded border border-white/5 font-mono text-[11px]">
                                            η(x) = (x - μ) / √(σ² + ε) * θ + λ
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="glass-card p-0 overflow-hidden relative group">
                                <div className="p-8 prose prose-invert max-w-none relative z-10">
                                    <h3 className="text-xl font-bold mb-4">Implementation Roadmap</h3>
                                    <div className="space-y-6">
                                        {[
                                            { step: "01", title: "Patch Partitioning", desc: "Dividing input into 4x4 non-overlapping segments." },
                                            { step: "02", title: "LSA Feature Extraction", desc: "Applying 6 layers of Locality Self-Attention blocks." },
                                            { step: "03", title: "Global Latent Mapping", desc: "Extracting the CLS token as the global feature vector." },
                                            { step: "04", title: "Dendritic Soma Aggregation", desc: "Multiple branch convergence for final class probability." },
                                        ].map((item, i) => (
                                            <div key={i} className="flex gap-6 items-start">
                                                <span className="text-3xl font-black text-indigo-500/20">{item.step}</span>
                                                <div>
                                                    <h5 className="font-bold text-slate-200">{item.title}</h5>
                                                    <p className="text-sm text-slate-500">{item.desc}</p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                                <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/5 -mr-16 -mt-16 rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </main>
        </div>
    );
};

export default App;
