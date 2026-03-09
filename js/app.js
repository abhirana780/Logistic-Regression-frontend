const API_BASE = "https://logistic-regression-backend.onrender.com";
let plotChart = null, importanceChart = null;
let lastTrainData = null;
let domainConfigs = {};
let currentKey = "hr_churn";
let lastTP = 0, lastFP = 0; // State for ROI updates

document.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch(`${API_BASE}/api/config`);
        domainConfigs = await res.json();
        initDomain();
    } catch(e) {
        console.error("Critical: Backend Down");
    }
});

// Navigation Logic
function switchPage(pId, btn) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
    
    // Deactivate all nav buttons
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    
    // Show target page
    const target = document.getElementById(pId);
    target.classList.remove('hidden');
    
    // If a button was passed (from sidebar), activate it
    if(btn) {
        btn.classList.add('active');
    } else {
        // Find the button in sidebar corresponding to this page index
        const index = parseInt(pId.split('-')[1]) - 1;
        document.querySelectorAll('.nav-btn')[index].classList.add('active');
    }
    
    // Scroll content to top
    document.querySelector('.content-area').scrollTop = 0;
}

function goToStep(stepNumber) {
    switchPage(`page-${stepNumber}`, null);
}

async function changeDomain() {
    currentKey = document.getElementById('master-ds-select').value;
    await fetch(`${API_BASE}/api/select_dataset`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({key: currentKey})
    });
    
    // Full State Reset
    lastTrainData = null;
    document.getElementById('dataset-container').classList.add('hidden');
    document.getElementById('ds-status-empty').classList.remove('hidden');
    document.getElementById('eval-overlay').classList.remove('hidden');
    document.getElementById('eval-content').classList.add('hidden');
    document.getElementById('pred-overlay').classList.remove('hidden');
    document.getElementById('predict-content').classList.add('hidden');
    document.getElementById('train-status').classList.add('hidden');
    
    initDomain();
    goToStep(1);
}

function initDomain() {
    const config = domainConfigs[currentKey];
    if(!config) return;
    
    document.getElementById('ds-title').innerText = config.name + " Intelligence";
    document.getElementById('ds-desc').innerText = `Strategic insights into ${config.name.toLowerCase()} trends.`;
    
    // Dynamic Training Checkboxes
    const container = document.getElementById('feature-cb-container');
    container.innerHTML = config.features.map(f => `
        <label class="feature-label">
            <input type="checkbox" class="feature-cb" value="${f}" checked>
            <span>${f.replace(/_/g, ' ').toUpperCase()}</span>
        </label>
    `).join('');

    // Update ROI Labels
    document.getElementById('lbl-saving-p').innerText = `Cost/Fail (${config.positive})`;
    document.getElementById('lbl-saving-s').innerText = `Cost/Action (${config.negative})`;
}

async function loadDataset() {
    const tableHead = document.getElementById('dataset-head');
    const tableBody = document.getElementById('dataset-body');
    const container = document.getElementById('dataset-container');
    const emptyState = document.getElementById('ds-status-empty');
    
    try {
        const res = await fetch(`${API_BASE}/api/dataset`);
        const json = await res.json();
        const config = json.config;
        
        tableHead.innerHTML = `<tr>${config.features.map(f => `<th>${f}</th>`).join('')}<th>${config.label_name}</th></tr>`;
        
        tableBody.innerHTML = json.data.slice(0, 15).map(r => {
            const config = domainConfigs[currentKey];
            const labelText = r.label ? config.positive.toUpperCase() : config.negative.toUpperCase();
            return `
                <tr>
                    ${config.features.map(f => `<td>${r[f]}</td>`).join('')}
                    <td class="bold ${r.label ? 'text-danger' : 'text-success'}">${labelText}</td>
                </tr>
            `;
        }).join('');
        
        emptyState.classList.add('hidden');
        container.classList.remove('hidden');
    } catch(e) { 
        alert("Sync Failed: Ensure backend server is running.");
    }
}

async function trainModel() {
    const btn = document.getElementById('train-btn');
    const selected = Array.from(document.querySelectorAll('.feature-cb:checked')).map(c => c.value);
    
    if (selected.length < 1) return alert("Strategic Error: Select at least 1 feature.");
    
    btn.innerText = "Processing Strategy...";
    btn.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/api/train`, {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ features: selected })
        });
        lastTrainData = await res.json();
        
        // Show components
        document.getElementById('eval-overlay').classList.add('hidden');
        document.getElementById('eval-content').classList.remove('hidden');
        document.getElementById('pred-overlay').classList.add('hidden');
        document.getElementById('predict-content').classList.remove('hidden');
        document.getElementById('train-status').classList.remove('hidden');
        
        updateMetrics();
        updateFormula(lastTrainData.coefficients);
        renderImportance(lastTrainData.importance);
        renderPlot(lastTrainData.plot);
        initPredictSliders(selected);
        
        btn.innerText = "Re-Train Strategy AI";
        btn.disabled = false;
    } catch(e) { 
        btn.innerText = "Logic Error"; 
        btn.disabled = false;
    }
}

function updateFormula(coefs) {
    const el = document.getElementById('dynamic-formula');
    let terms = Object.entries(coefs.weights).map(([f, w]) => 
        ` (${w.toFixed(2)} * ${f.replace(/_/g, ' ').toUpperCase()})`
    ).join(' + ');
    el.innerHTML = `P = Sigmoid( ${coefs.intercept.toFixed(2)} + ${terms} )`;
}

function updateMetrics() {
    if (!lastTrainData) return;
    const thresh = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-val').innerText = thresh.toFixed(2);
    
    const { y_true, y_probs } = lastTrainData.test_results;
    let tp=0, tn=0, fp=0, fn=0;
    y_probs.forEach((p, i) => {
        const pred = p >= thresh ? 1 : 0;
        const actual = y_true[i];
        if(pred===1 && actual===1) tp++;
        else if(pred===0 && actual===0) tn++;
        else if(pred===1 && actual===0) fp++;
        else fn++;
    });
    
    document.getElementById('metric-acc').innerText = `${((tp + tn) / y_true.length * 100).toFixed(1)}%`;
    const recall = (tp / (tp + fn)) * 100 || 0;
    document.getElementById('metric-rec').innerText = `${recall.toFixed(1)}%`;
    lastTP = tp; 
    lastFP = fp;
    updateROI();
}

function updateROI() {
    if (!lastTrainData) return;
    const tp = lastTP;
    const fp = lastFP;
    const costMiss = parseFloat(document.getElementById('cost-churn').value) || 0;
    const costAction = parseFloat(document.getElementById('cost-save').value) || 0;
    const savings = (tp * costMiss) - ((tp + fp) * costAction);
    const el = document.getElementById('total-savings');
    el.innerText = `$${savings.toLocaleString()}`;
    el.className = `savings-value ${savings >= 0 ? 'text-success' : 'text-danger'}`;
    
    // Model Health Check
    const acc = parseFloat(document.getElementById('metric-acc').innerText);
    const health = document.getElementById('model-health-tag') || { innerText: "" };
    if (acc > 85) { health.innerText = "Elite Accuracy"; health.style.color = "var(--success)"; }
    else if (acc > 70) { health.innerText = "Reliable Model"; health.style.color = "var(--primary)"; }
    else { health.innerText = "Under-Trained"; health.style.color = "var(--danger)"; }
}

function renderImportance(imp) {
    const ctx = document.getElementById('importanceChart').getContext('2d');
    if(importanceChart) importanceChart.destroy();
    
    importanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: imp.map(i => i.feature.replace(/_/g, ' ').toUpperCase()),
          datasets: [{ 
              label: 'Strategic Weight', 
              data: imp.map(i => i.weight), 
              backgroundColor: '#4318ff',
              borderRadius: { topLeft: 0, bottomLeft: 0, topRight: 10, bottomRight: 10 },
              borderSkipped: false,
              barThickness: 26
          }]
        },
        options: { 
            indexAxis: 'y', 
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { 
                x: { display: false }, 
                y: { 
                    grid: { display: false },
                    ticks: {
                        color: '#2b3674',
                        padding: 10,
                        font: { family: 'Outfit', size: 11, weight: '700' },
                        autoSkip: false
                    }
                } 
            },
            layout: { padding: { left: 100, right: 40 } }
        }
    });
}

function renderPlot(plot) {
    const ctx = document.getElementById('evalChart').getContext('2d');
    if(plotChart) plotChart.destroy();
    plotChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Negative Class', data: plot.points.filter(p=>p.actual===0), backgroundColor: '#a3aed0', pointRadius: 5 },
                { label: 'Potential Risk', data: plot.points.filter(p=>p.actual===1), backgroundColor: '#ee5d50', pointRadius: 7, pointStyle: 'rectRot' }
            ]
        },
        options: { 
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { font: { family: 'Outfit', weight: '700' } } } },
            scales: { 
                x: { 
                    grid: { color: '#e9edf7' },
                    title: { display: true, text: plot.x_label.toUpperCase(), color: '#707eae', font: { weight: '800' } } 
                }, 
                y: { 
                    grid: { color: '#e9edf7' },
                    title: { display: true, text: plot.y_label.toUpperCase(), color: '#707eae', font: { weight: '800' } } 
                } 
            }
        }
    });
}

function initPredictSliders(feats) {
    const container = document.getElementById('dynamic-sliders');
    container.innerHTML = feats.map(f => {
        let min=0, max=10, step=0.1;
        if(f.includes('hours') || f.includes('income') || f.includes('amount')) { max=500; step=1; if(f.includes('income')) max=100000; }
        else if(f.includes('age') || f.includes('tenure') || f.includes('years')) { max=20; step=1; }
        
        return `
            <div class="slider-group">
                <div class="slider-header">
                    <label>${f.replace(/_/g, ' ').toUpperCase()}</label>
                    <span class="val" id="val-${f}-text">${max/2}</span>
                </div>
                <input type="range" class="pred-slider" data-feat="${f}" min="${min}" max="${max}" step="${step}" value="${max/2}" oninput="runLivePredict()">
            </div>
        `;
    }).join('');
    runLivePredict();
}

async function runLivePredict() {
    const sliders = document.querySelectorAll('.pred-slider');
    const values = Array.from(sliders).map(s => {
        document.getElementById(`val-${s.dataset.feat}-text`).innerText = s.value;
        return parseFloat(s.value);
    });
    
    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({values})
        });
        const { probability } = await res.json();
        const probPct = Math.round(probability * 100);
        
        document.getElementById('out-prob').innerText = `${probPct}%`;
        document.getElementById('risk-fill').style.height = `${probPct}%`;
        
        // --- NEW: Factor Contribution Breakdown ---
        const contContainer = document.getElementById('contribution-list');
        const weights = lastTrainData.coefficients.weights;
        const intercept = lastTrainData.coefficients.intercept;
        
        contContainer.innerHTML = '';
        Object.entries(weights).forEach(([f, w], idx) => {
            const val = values[idx];
            const contribution = w * val;
            const absCont = Math.min(Math.abs(contribution) * 10, 100); // Scale for visual
            
            const item = document.createElement('div');
            item.className = 'contribution-item';
            item.innerHTML = `
                <span class="contribution-name">${f.replace(/_/g, ' ').toUpperCase()}</span>
                <div class="contribution-bar-bg">
                    <div class="contribution-fill ${contribution > 0 ? 'pos-influence' : 'neg-influence'}" 
                         style="width: ${absCont}%"></div>
                </div>
                <span class="contribution-val">${contribution > 0 ? '+' : ''}${contribution.toFixed(1)}</span>
            `;
            contContainer.appendChild(item);
        });
        
        const config = domainConfigs[currentKey];
        const status = document.getElementById('out-class');
        const advice = document.getElementById('out-advice');
        
        status.innerText = probPct > 50 ? `PREDICTED ${config.positive.toUpperCase()}` : `STATUS ${config.negative.toUpperCase()}`;
        status.style.background = probPct > 50 ? 'var(--danger)' : 'var(--success)';
        
        // --- Conclusive Strategic Verdict ---
        const conclusion = document.getElementById('final-conclusion');
        const actionPlan = document.getElementById('action-plan');
        
        if (probPct > 80) {
            conclusion.innerText = `CRITICAL RISK: Mathematical certainty of ${config.positive} is extremely high. Immediate executive intervention is mandatory to mitigate loss.`;
            actionPlan.innerText = `Emergency ${config.positive} Prevention Protocol`;
            actionPlan.className = "text-sm bold text-danger";
        } else if (probPct > 50) {
            conclusion.innerText = `MODERATE WARNING: The profile shows patterns transitioning towards ${config.positive}. Closely monitor these variables over the next quarter.`;
            actionPlan.innerText = `Standard Risk Mitigation / Observation`;
            actionPlan.className = "text-sm bold text-warning";
        } else {
            conclusion.innerText = `STRATEGIC STABILITY: The current scenario aligns with the ${config.negative} cohort. No immediate structural changes are recommended.`;
            actionPlan.innerText = `Maintain Current Operational Strategy`;
            actionPlan.className = "text-sm bold text-success";
        }
    } catch(e) {}
}

function copyStrategicReport() {
    const prob = document.getElementById('out-prob').innerText;
    const conclusion = document.getElementById('final-conclusion').innerText;
    const action = document.getElementById('action-plan').innerText;
    const text = `STRATEGIC AI REPORT\nProbability: ${prob}\nVerdict: ${conclusion}\nAction Plan: ${action}`;
    
    navigator.clipboard.writeText(text).then(() => {
        alert("📊 Report Copied to Clipboard! You can now paste this into your MBA Case Study.");
    });
}
