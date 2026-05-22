document.addEventListener('DOMContentLoaded', () => {

    // ── Element refs ────────────────────────────────────────────────────────
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const predictBtn = document.getElementById('predict-btn');
    const loadingOverlay = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const debugBar = document.getElementById('debug-bar');

    const resultOrig = document.getElementById('result-orig');
    const resultMask = document.getElementById('result-mask');
    const resultOverlay = document.getElementById('result-overlay');
    const resultHeatmap = document.getElementById('result-heatmap');
    const resultUncertainty = document.getElementById('result-uncertainty');

    const fftPlaceholder = document.getElementById('fft-placeholder');
    const uncPlaceholder = document.getElementById('unc-placeholder');

    let currentFile = null;
    function dbg(msg) {
        console.log('[ESCNet]', msg);
        debugBar.style.display = 'block';
        debugBar.textContent += msg + '\n';
        debugBar.scrollTop = debugBar.scrollHeight;
    }
    function showImage(imgEl, placeholder, b64) {
        return new Promise((resolve, reject) => {
            if (!b64) { reject(new Error('base64 string is empty/null')); return; }
            imgEl.onload = () => resolve();
            imgEl.onerror = () => reject(new Error('browser could not decode image'));
            imgEl.src = `data:image/png;base64,${b64}`;
            imgEl.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
        });
    }
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) { alert('Please select an image file.'); return; }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        debugBar.textContent = '';
        debugBar.style.display = 'none';

        [resultHeatmap, resultUncertainty].forEach(el => { el.style.display = 'none'; el.src = ''; });
        if (fftPlaceholder) fftPlaceholder.style.display = 'flex';
        if (uncPlaceholder) uncPlaceholder.style.display = 'flex';

        predictBtn.disabled = true;
        predictBtn.querySelector('span').textContent = 'Analyzing…';
        loadingOverlay.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            dbg('POST /predict started');
            const response = await fetch('/predict', { method: 'POST', body: formData });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server error ${response.status}`);
            }

            const data = await response.json();
            dbg(`Response keys: ${JSON.stringify(Object.keys(data))}`);
            Object.keys(data).forEach(k => {
                dbg(`  ${k}: ${typeof data[k] === 'string' ? data[k].length + ' chars' : data[k]}`);
            });

            await showImage(resultOrig, null, data.original);
            await showImage(resultMask, null, data.mask);
            if (data.overlay) {
                await showImage(resultOverlay, null, data.overlay);
            }
            
            if (data.heatmap) {
                try { await showImage(resultHeatmap, fftPlaceholder, data.heatmap); dbg('FFT: OK'); }
                catch (e) { dbg('FFT ERROR: ' + e.message); if (fftPlaceholder) fftPlaceholder.textContent = 'FFT failed'; }
            } else { dbg('FFT: KEY MISSING'); if (fftPlaceholder) fftPlaceholder.textContent = 'Not returned by server'; }
            
            if (data.uncertainty) {
                dbg('Uncertainty base64 length: ' + data.uncertainty.length);
                try { await showImage(resultUncertainty, uncPlaceholder, data.uncertainty); dbg('Uncertainty: OK ✓'); }
                catch (e) { dbg('Uncertainty ERROR: ' + e.message); if (uncPlaceholder) uncPlaceholder.textContent = 'Decode failed'; }
            } else { dbg('Uncertainty: KEY MISSING — check app.py terminal'); if (uncPlaceholder) uncPlaceholder.textContent = 'Not returned by server'; }
            
            loadingOverlay.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            previewContainer.classList.add('hidden');
            dropZone.classList.remove('hidden');
            window.scrollTo({ top: resultsSection.offsetTop - 50, behavior: 'smooth' });
            currentFile = null;

        } catch (err) {
            dbg('FETCH ERROR: ' + err.message);
            alert('Error: ' + err.message);
            loadingOverlay.classList.add('hidden');
        } finally {
            predictBtn.disabled = false;
            predictBtn.querySelector('span').textContent = 'Analyze with ESCNet';
        }
    });

    // ── End ──
});
