/* ── DOM refs ── */
const dropZone          = document.getElementById('dropZone');
const fileInput         = document.getElementById('fileInput');
const browseBtn         = document.getElementById('browseBtn');
const previewWrap       = document.getElementById('previewWrap');
const videoPreview      = document.getElementById('videoPreview');
const fileInfo          = document.getElementById('fileInfo');
const analyzeBtn        = document.getElementById('analyzeBtn');
const generateVideoBtn  = document.getElementById('generateVideoBtn');
const changeBtn         = document.getElementById('changeBtn');
const statusBar         = document.getElementById('statusBar');
const statusText        = document.getElementById('statusText');
const resultsCard       = document.getElementById('resultsCard');
const resultsBody       = document.getElementById('resultsBody');
const resultsFooter     = document.getElementById('resultsFooter');
const copyBtn           = document.getElementById('copyBtn');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
const videoResultCard   = document.getElementById('videoResultCard');
const videoResultPlayer = document.getElementById('videoResultPlayer');
const videoResultBadge  = document.getElementById('videoResultBadge');
const downloadVideoBtn  = document.getElementById('downloadVideoBtn');
const errorBanner       = document.getElementById('errorBanner');
const errorText         = document.getElementById('errorText');

let selectedFile = null;
let rawMarkdown  = '';

/* ── Drag & drop ── */
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
['dragleave', 'dragend'].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'))
);
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── File handling ── */
function handleFile(file) {
  const allowed = ['video/mp4','video/quicktime','video/avi','video/webm','video/x-matroska','video/x-msvideo'];
  const ext = file.name.split('.').pop().toLowerCase();
  const allowedExt = ['mp4','mov','avi','webm','mkv'];

  if (!allowedExt.includes(ext)) {
    showError('Unsupported format. Please upload MP4, MOV, AVI, WebM, or MKV.');
    return;
  }
  if (file.size > 150 * 1024 * 1024) {
    showError('File is too large. Maximum size is 150 MB.');
    return;
  }

  selectedFile = file;
  hideError();
  resetResults();

  // Show preview
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  fileInfo.textContent = `${file.name}  ·  ${formatBytes(file.size)}`;
  dropZone.classList.add('hidden');
  previewWrap.classList.remove('hidden');
}

changeBtn.addEventListener('click', () => {
  resetAll();
});

/* ── Analyze ── */
analyzeBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  startAnalysis();
});

/* ── Generate annotated video ── */
generateVideoBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  startVideoGeneration();
});

function startVideoGeneration() {
  setAllButtonsDisabled(true);
  hideError();
  videoResultCard.classList.add('hidden');

  statusBar.classList.remove('hidden');
  statusText.textContent = 'Uploading video for annotation...';

  const formData = new FormData();
  formData.append('video', selectedFile);

  fetch('/generate-video', { method: 'POST', body: formData })
    .then(res => {
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      function pump() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            statusBar.classList.add('hidden');
            setAllButtonsDisabled(false);
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          lines.forEach(line => {
            if (!line.startsWith('data: ')) return;
            try {
              const payload = JSON.parse(line.slice(6));
              handleVideoPayload(payload);
            } catch (_) {}
          });

          return pump();
        });
      }
      return pump();
    })
    .catch(err => {
      showError('Network error: ' + err.message);
      statusBar.classList.add('hidden');
      setAllButtonsDisabled(false);
    });
}

function handleVideoPayload(p) {
  if (p.error) {
    showError(p.error);
    statusBar.classList.add('hidden');
    setAllButtonsDisabled(false);
    return;
  }

  if (p.status) {
    statusText.textContent = p.status;
    return;
  }

  if (p.progress !== undefined) {
    statusText.textContent = `Annotating video... ${p.progress.toFixed(0)}%`;
    return;
  }

  if (p.done && p.vid_id) {
    const downloadUrl = `/download/${p.vid_id}`;
    downloadVideoBtn.href = downloadUrl;

    // Also show inline player — fetch as blob so it plays before the
    // download link expires (the /download route removes it from memory)
    fetch(downloadUrl)
      .then(r => r.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        videoResultPlayer.src = url;
        downloadVideoBtn.href = url;
        downloadVideoBtn.download = 'annotated_technique.mp4';
        videoResultBadge.textContent = selectedFile ? selectedFile.name.replace(/\.[^.]+$/, '') + '_visualized' : '';
        videoResultCard.classList.remove('hidden');
        videoResultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      })
      .catch(() => {
        // Blob fetch failed (already consumed) — just open download directly
        window.location.href = downloadUrl;
      });

    statusBar.classList.add('hidden');
    setAllButtonsDisabled(false);
  }
}

function setAllButtonsDisabled(disabled) {
  analyzeBtn.disabled        = disabled;
  generateVideoBtn.disabled  = disabled;
  changeBtn.disabled         = disabled;
}

function startAnalysis() {
  analyzeBtn.disabled = true;
  changeBtn.disabled  = true;
  hideError();
  resetResults();

  statusBar.classList.remove('hidden');
  statusText.textContent = 'Uploading video…';

  const formData = new FormData();
  formData.append('video', selectedFile);

  rawMarkdown = '';

  // Use fetch with ReadableStream for SSE
  fetch('/analyze', { method: 'POST', body: formData })
    .then(res => {
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      function pump() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            finishAnalysis();
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // keep incomplete line

          lines.forEach(line => {
            if (!line.startsWith('data: ')) return;
            try {
              const payload = JSON.parse(line.slice(6));
              handlePayload(payload);
            } catch (_) {}
          });

          return pump();
        });
      }
      return pump();
    })
    .catch(err => {
      showError('Network error: ' + err.message);
      resetButtons();
    });
}

function handlePayload(p) {
  if (p.error) {
    showError(p.error);
    statusBar.classList.add('hidden');
    resetButtons();
    return;
  }

  if (p.status) {
    statusText.textContent = p.status;
    return;
  }

  if (p.start) {
    statusText.textContent = 'Generating coaching report…';
    resultsCard.classList.remove('hidden');
    resultsBody.classList.add('streaming');
    return;
  }

  if (p.chunk) {
    rawMarkdown += p.chunk;
    resultsBody.innerHTML = marked.parse(rawMarkdown);
    // auto-scroll if near bottom
    const el = resultsBody;
    if (el.scrollHeight - el.scrollTop < el.clientHeight + 120) {
      resultsCard.scrollIntoView({ block: 'end', behavior: 'smooth' });
    }
    return;
  }

  if (p.done) {
    finishAnalysis();
  }
}

function finishAnalysis() {
  statusBar.classList.add('hidden');
  resultsBody.classList.remove('streaming');
  resultsFooter.classList.remove('hidden');
  resetButtons();
  resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Copy report ── */
copyBtn.addEventListener('click', () => {
  if (!rawMarkdown) return;
  navigator.clipboard.writeText(rawMarkdown).then(() => {
    const orig = copyBtn.textContent;
    copyBtn.textContent = '✅ Copied!';
    setTimeout(() => { copyBtn.textContent = orig; }, 2000);
  });
});

/* ── Analyze another ── */
analyzeAnotherBtn.addEventListener('click', resetAll);

/* ── Helpers ── */
function resetAll() {
  selectedFile = null;
  rawMarkdown  = '';
  fileInput.value = '';
  videoPreview.src = '';
  videoResultPlayer.src = '';
  previewWrap.classList.add('hidden');
  videoResultCard.classList.add('hidden');
  dropZone.classList.remove('hidden');
  resetResults();
  hideError();
  resetButtons();
  statusBar.classList.add('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function resetResults() {
  resultsCard.classList.add('hidden');
  resultsBody.innerHTML = '';
  resultsBody.classList.remove('streaming');
  resultsFooter.classList.add('hidden');
  rawMarkdown = '';
}

function resetButtons() {
  setAllButtonsDisabled(false);
}

function showError(msg) {
  errorText.textContent = msg;
  errorBanner.classList.remove('hidden');
  errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
  errorBanner.classList.add('hidden');
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
