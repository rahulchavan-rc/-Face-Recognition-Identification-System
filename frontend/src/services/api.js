const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = typeof data.detail === 'string' ? data.detail : 'The service request failed.';
    throw new Error(message);
  }

  return data;
}

export const api = {
  getStatus: () => request('/status'),
  getConfig: () => request('/config'),
  getPeople: () => request('/people'),
  getRecognitionHistory: (limit = 100) => request(`/recognition/history?limit=${limit}`),
  getEvaluation: () => request('/evaluation'),
  recognize: (file, threshold, source) => {
    if (!source) {
      throw new Error('Recognition source is required.');
    }
    const formData = new FormData();
    formData.append('file', file);
    const params = new URLSearchParams();
    if (threshold != null) params.set('threshold', String(threshold));
    if (source) params.set('source', source);
    const query = params.toString() ? `?${params.toString()}` : '';
    return request(`/recognize${query}`, { method: 'POST', body: formData });
  },
  enroll: (name, files) => {
    const formData = new FormData();
    formData.append('name', name);
    files.forEach((file) => formData.append('files', file));
    return request('/enroll', { method: 'POST', body: formData });
  },
  updateThreshold: (value) => request(`/config/threshold?value=${encodeURIComponent(value)}`, { method: 'PATCH' }),
  deletePerson: (person) => request(`/people/${encodeURIComponent(person)}`, { method: 'DELETE' }),
};

export { API_BASE_URL };
