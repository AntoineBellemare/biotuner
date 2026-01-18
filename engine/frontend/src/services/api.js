/**
 * API Client for Biotuner Backend
 * Handles all HTTP requests and WebSocket connections
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class BiotunerAPI {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    this.ws = null;
    this.wsCallbacks = {};
  }

  // ============================================================================
  // File Upload & Processing
  // ============================================================================

  async getIntervalCatalog() {
    const response = await this.client.get('/api/interval-catalog');
    return response.data;
  }

  async uploadFile(file, sessionId = null) {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) {
      formData.append('session_id', sessionId);
    }

    const response = await this.client.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async selectColumn(sessionId, columnIndex) {
    const response = await this.client.post('/api/select-column', null, {
      params: { session_id: sessionId, column_index: columnIndex }
    });
    return response.data;
  }

  // ============================================================================
  // Biotuner Analysis
  // ============================================================================

  async analyze(config) {
    const response = await this.client.post('/api/analyze', config);
    return response.data;
  }

  async reduceTuning(sessionId, nSteps = 12, maxRatio = 2.0) {
    const response = await this.client.post('/api/tuning-reduction', {
      session_id: sessionId,
      n_steps: nSteps,
      max_ratio: maxRatio,
    });
    return response.data;
  }

  async playTuning(sessionId, tuning, baseFreq = 120, duration = 0.5) {
    const response = await this.client.post('/api/play-tuning', {
      session_id: sessionId,
      tuning,
      base_freq: baseFreq,
      duration,
    }, {
      responseType: 'blob',
    });
    return response.data;
  }

  // ============================================================================
  // Chord Generation
  // ============================================================================

  async generateChords(config) {
    const response = await this.client.post('/api/generate-chords', config);
    return response.data;
  }

  async getChordAudio(sessionId, tuning, numChords = 3, baseFreq = 440, duration = 1.0) {
    const response = await this.client.post('/api/chord-audio', {
      session_id: sessionId,
      tuning,
      num_chords: numChords,
      base_freq: baseFreq,
      duration,
    }, {
      responseType: 'blob',
    });
    return response.data;
  }

  async exportMidi(sessionId, chords, boundTimes, totalDuration = null) {
    const response = await this.client.post('/api/export-midi', {
      session_id: sessionId,
      chords,
      bound_times: boundTimes,
      total_duration: totalDuration,
    }, {
      responseType: 'blob',
    });
    return response.data;
  }

  // ============================================================================
  // Biocolors
  // ============================================================================

  async generateBiocolors(config) {
    const response = await this.client.post('/api/biocolors', config);
    return response.data;
  }

  async exportPalette(format, colors, filename = 'palette') {
    const response = await this.client.post(`/api/export-palette/${format}`, {
      colors,
      filename,
    }, {
      responseType: 'blob',
    });
    return response.data;
  }

  // ============================================================================
  // Session Management
  // ============================================================================

  async getSession(sessionId) {
    const response = await this.client.get(`/api/session/${sessionId}`);
    return response.data;
  }

  async deleteSession(sessionId) {
    const response = await this.client.delete(`/api/session/${sessionId}`);
    return response.data;
  }

  // ============================================================================
  // WebSocket Connection
  // ============================================================================

  connectWebSocket(sessionId, callbacks = {}) {
    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/${sessionId}`;
    
    this.ws = new WebSocket(wsUrl);
    this.wsCallbacks = callbacks;

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      if (callbacks.onOpen) callbacks.onOpen();
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (callbacks.onMessage) callbacks.onMessage(data);
      
      // Handle specific message types
      if (data.type === 'progress' && callbacks.onProgress) {
        callbacks.onProgress(data);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (callbacks.onError) callbacks.onError(error);
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
      if (callbacks.onClose) callbacks.onClose();
    };
  }

  sendWebSocketMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  closeWebSocket() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  async getInfo() {
    const response = await this.client.get('/api/info');
    return response.data;
  }

  // Helper to download blob as file
  downloadBlob(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }
}

// Export singleton instance
const apiClient = new BiotunerAPI();
export default apiClient;
