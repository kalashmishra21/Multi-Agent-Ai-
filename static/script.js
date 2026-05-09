/**
 * AI Study Planner — Frontend Script
 * Handles form submission, SSE streaming, and UI updates.
 */

// ─── DOM References ───────────────────────────────────────────────────────────
const form          = document.getElementById('studyForm');
const generateBtn   = document.getElementById('generateBtn');
const pipeline      = document.getElementById('pipeline');
const outputPlaceholder = document.getElementById('outputPlaceholder');
const outputContent = document.getElementById('outputContent');
const outputBody    = document.getElementById('outputBody');
const errorBox      = document.getElementById('errorBox');
const errorMessage  = document.getElementById('errorMessage');

// ─── State ────────────────────────────────────────────────────────────────────
let eventSource = null;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Set an agent step's visual state.
 * @param {number} agentNum - 1 to 4
 * @param {'waiting'|'active'|'done'} state
 * @param {string} statusText
 */
function setAgentState(agentNum, state, statusText) {
  const step      = document.getElementById(`agent-${agentNum}`);
  const statusEl  = document.getElementById(`status-${agentNum}`);

  step.classList.remove('active', 'done');
  if (state === 'active') step.classList.add('active');
  if (state === 'done')   step.classList.add('done');

  if (statusText) statusEl.textContent = statusText;
}

/** Reset all agent steps to waiting state. */
function resetAgents() {
  for (let i = 1; i <= 4; i++) {
    setAgentState(i, 'waiting', 'Waiting...');
  }
}

/** Show the error box with a message. */
function showError(msg) {
  errorBox.style.display    = 'flex';
  errorMessage.textContent  = msg;
  outputPlaceholder.style.display = 'none';
  outputContent.style.display     = 'none';
}

/** Hide the error box. */
function hideError() {
  errorBox.style.display = 'none';
}

/** Set the generate button to loading state. */
function setLoading(loading) {
  if (loading) {
    generateBtn.disabled = true;
    generateBtn.innerHTML = `<span class="spinner"></span><span class="btn-text">Generating...</span>`;
  } else {
    generateBtn.disabled = false;
    generateBtn.innerHTML = `<span class="btn-icon">⚡</span><span class="btn-text">Generate My Study Plan</span>`;
  }
}

/**
 * Format the raw plan text into readable HTML.
 * Highlights section headers and separators.
 * @param {string} text
 * @returns {string} HTML string
 */
function formatOutput(text) {
  // Escape HTML
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Bold section headers like [SECTION NAME] or ALL CAPS lines ending with :
  html = html.replace(/^\[(.+?)\]$/gm, '<strong style="color:#4f46e5;font-size:0.95rem;">[$1]</strong>');
  html = html.replace(/^(={3,}.*={3,})$/gm, '<span style="color:#6366f1;font-weight:700;">$1</span>');
  html = html.replace(/^(-{3,})$/gm, '<span style="color:#e2e8f0;">$1</span>');

  // Bold Day X lines
  html = html.replace(/(Day \d+[^:]*:)/g, '<strong style="color:#1e293b;">$1</strong>');

  // Bold Week X lines
  html = html.replace(/(Week \d+[^:]*:)/g, '<strong style="color:#4f46e5;">$1</strong>');

  return html;
}

/** Copy output text to clipboard. */
function copyOutput() {
  const text = outputBody.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
  }).catch(() => {
    alert('Could not copy. Please select and copy manually.');
  });
}

// Make copyOutput globally accessible (called from HTML onclick)
window.copyOutput = copyOutput;

// ─── Form Submit ──────────────────────────────────────────────────────────────

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  // Close any existing SSE connection
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  // Collect form data
  const payload = {
    subject:        document.getElementById('subject').value.trim(),
    goal:           document.getElementById('goal').value.trim(),
    hours_per_day:  document.getElementById('hours_per_day').value.trim(),
    deadline:       document.getElementById('deadline').value.trim(),
    learning_style: document.getElementById('learning_style').value.trim(),
  };

  // Basic client-side validation
  for (const [key, val] of Object.entries(payload)) {
    if (!val) {
      showError(`Please fill in all fields. Missing: ${key.replace('_', ' ')}`);
      return;
    }
  }

  // Reset UI
  hideError();
  setLoading(true);
  resetAgents();
  pipeline.style.display = 'block';
  outputPlaceholder.style.display = 'none';
  outputContent.style.display     = 'none';
  outputBody.innerHTML = '';

  // Send request to backend
  let response;
  try {
    response = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    setLoading(false);
    showError('Network error. Make sure the Flask server is running.');
    return;
  }

  if (!response.ok) {
    const errData = await response.json().catch(() => ({}));
    setLoading(false);
    showError(errData.error || 'Server error. Please try again.');
    return;
  }

  // ─── SSE Stream Handling ───────────────────────────────────────────────────
  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer    = '';

  const processChunk = (chunk) => {
    buffer += chunk;
    const parts = buffer.split('\n\n');
    buffer = parts.pop(); // keep incomplete last part

    for (const part of parts) {
      if (!part.trim()) continue;

      // Parse event name
      const eventMatch = part.match(/^event:\s*(.+)$/m);
      const dataMatch  = part.match(/^data:\s*(.+)$/m);

      if (!eventMatch || !dataMatch) continue;

      const eventName = eventMatch[1].trim();
      let data;
      try {
        data = JSON.parse(dataMatch[1]);
      } catch {
        continue;
      }

      handleSSEEvent(eventName, data);
    }
  };

  const handleSSEEvent = (eventName, data) => {
    switch (eventName) {

      case 'agent_start':
        setAgentState(data.agent, 'active', data.message);
        break;

      case 'agent_done':
        setAgentState(data.agent, 'done', 'Completed ✓');
        break;

      case 'complete':
        setLoading(false);
        outputContent.style.display = 'block';
        outputBody.innerHTML = formatOutput(data.final_output);
        // Scroll output into view on mobile
        outputContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
        break;

      case 'error':
        setLoading(false);
        showError(data.message);
        break;
    }
  };

  // Read the stream
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      processChunk(decoder.decode(value, { stream: true }));
    }
  } catch (err) {
    setLoading(false);
    showError('Stream interrupted. Please try again.');
  }
});

// ─── Subject Autocomplete ─────────────────────────────────────────────────────

const SUBJECTS = [
  { icon: '🤖', text: 'Machine Learning' },
  { icon: '🧠', text: 'Deep Learning' },
  { icon: '📊', text: 'Data Science' },
  { icon: '🐍', text: 'Python Programming' },
  { icon: '🌐', text: 'Web Development' },
  { icon: '⚛️', text: 'React.js' },
  { icon: '🟨', text: 'JavaScript' },
  { icon: '☕', text: 'Java Programming' },
  { icon: '➕', text: 'C++ Programming' },
  { icon: '🗄️', text: 'Database Management (SQL)' },
  { icon: '🔗', text: 'Data Structures and Algorithms' },
  { icon: '☁️', text: 'Cloud Computing (AWS)' },
  { icon: '🐳', text: 'Docker and Kubernetes' },
  { icon: '🔒', text: 'Cybersecurity' },
  { icon: '📱', text: 'Android Development' },
  { icon: '🍎', text: 'iOS Development (Swift)' },
  { icon: '🔥', text: 'Node.js' },
  { icon: '🎨', text: 'UI/UX Design' },
  { icon: '📈', text: 'Data Analysis' },
  { icon: '🧮', text: 'Mathematics for AI' },
  { icon: '📉', text: 'Statistics' },
  { icon: '🔤', text: 'Natural Language Processing' },
  { icon: '👁️', text: 'Computer Vision' },
  { icon: '🕸️', text: 'Django / Flask' },
  { icon: '🦀', text: 'Rust Programming' },
  { icon: '🐹', text: 'Go (Golang)' },
  { icon: '📦', text: 'DevOps' },
  { icon: '🧬', text: 'Bioinformatics' },
  { icon: '💹', text: 'Blockchain Development' },
  { icon: '🎮', text: 'Game Development (Unity)' },
  { icon: '🤝', text: 'System Design' },
  { icon: '🧩', text: 'Operating Systems' },
  { icon: '📡', text: 'Computer Networks' },
  { icon: '🏗️', text: 'Software Engineering' },
  { icon: '🔬', text: 'Artificial Intelligence' },
];

const subjectInput   = document.getElementById('subject');
const suggestionsList = document.getElementById('suggestionsList');
let activeIndex = -1;

/**
 * Highlight matching part of text.
 * @param {string} text - Full suggestion text
 * @param {string} query - User typed query
 * @returns {string} HTML with matched part wrapped in <mark>
 */
function highlightMatch(text, query) {
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return text;
  return (
    text.slice(0, idx) +
    `<mark>${text.slice(idx, idx + query.length)}</mark>` +
    text.slice(idx + query.length)
  );
}

/** Render suggestion dropdown. */
function renderSuggestions(query) {
  const q = query.trim().toLowerCase();
  if (!q) {
    closeSuggestions();
    return;
  }

  const filtered = SUBJECTS.filter(s => s.text.toLowerCase().includes(q));

  if (filtered.length === 0) {
    closeSuggestions();
    return;
  }

  suggestionsList.innerHTML = filtered.map((s, i) => `
    <li data-index="${i}" data-value="${s.text}">
      <span class="suggestion-icon">${s.icon}</span>
      <span class="suggestion-text">${highlightMatch(s.text, query.trim())}</span>
    </li>
  `).join('');

  activeIndex = -1;
  suggestionsList.classList.add('open');

  // Click handler for each item
  suggestionsList.querySelectorAll('li').forEach(li => {
    li.addEventListener('mousedown', (e) => {
      e.preventDefault(); // prevent blur before click
      subjectInput.value = li.dataset.value;
      closeSuggestions();
    });
  });
}

/** Close and clear the suggestions dropdown. */
function closeSuggestions() {
  suggestionsList.classList.remove('open');
  suggestionsList.innerHTML = '';
  activeIndex = -1;
}

/** Move active highlight up/down in list. */
function moveActive(direction) {
  const items = suggestionsList.querySelectorAll('li');
  if (!items.length) return;

  items.forEach(li => li.classList.remove('active'));
  activeIndex += direction;

  if (activeIndex < 0) activeIndex = items.length - 1;
  if (activeIndex >= items.length) activeIndex = 0;

  items[activeIndex].classList.add('active');
  items[activeIndex].scrollIntoView({ block: 'nearest' });
}

// Input event — show suggestions as user types
subjectInput.addEventListener('input', () => {
  renderSuggestions(subjectInput.value);
});

// Keyboard navigation
subjectInput.addEventListener('keydown', (e) => {
  if (!suggestionsList.classList.contains('open')) return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    moveActive(1);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    moveActive(-1);
  } else if (e.key === 'Enter') {
    const active = suggestionsList.querySelector('li.active');
    if (active) {
      e.preventDefault();
      subjectInput.value = active.dataset.value;
      closeSuggestions();
    }
  } else if (e.key === 'Escape') {
    closeSuggestions();
  }
});

// Close when clicking outside
document.addEventListener('click', (e) => {
  if (!subjectInput.contains(e.target) && !suggestionsList.contains(e.target)) {
    closeSuggestions();
  }
});

// Show all on focus if input has value
subjectInput.addEventListener('focus', () => {
  if (subjectInput.value.trim()) {
    renderSuggestions(subjectInput.value);
  }
});
