import { useEffect, useRef, useState } from 'react';
import {
  ArrowRight,
  CheckCircle2,
  FileImage,
  Gauge,
  Layers3,
  LayoutDashboard,
  Users,
  Camera,
  History,
  Settings,
  Cpu,
  Bell,
  Search,
  Moon,
  Sun,
  Upload,
  XCircle,
  CircleDashed,
} from 'lucide-react';
import { api } from './services/api';

function getErrorMessage(error, fallback = 'Request failed') {
  if (typeof error === 'string') return error;
  if (error?.detail) {
    return typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail);
  }
  if (error?.message) return error.message;
  return fallback;
}

function Navbar({ onLaunchRecognition, onWorkspaceSearch, theme, onToggleTheme }) {
  const [searchValue, setSearchValue] = useState('');

  const submitSearch = (event) => {
    event.preventDefault();
    if (!searchValue.trim()) return;
    onWorkspaceSearch(searchValue);
  };

  return (
    <>
      <aside className="sidebar">
        <div className="sidebar-brand brand-wrap">
          <div className="brand-mark">F</div>
          <div><strong>FaceSecure AI</strong><small>Identity security</small></div>
        </div>
        <div className="sidebar-group">
          <span className="sidebar-kicker">Workspace</span>
          <a href="#dashboard"><LayoutDashboard size={16} /> Dashboard</a>
          <a href="#people"><Users size={16} /> People</a>
          <a href="#recognition"><Camera size={16} /> Live recognition</a>
          <a href="#login-details"><History size={16} /> Login details</a>
          <a href="#platform-readiness"><CheckCircle2 size={16} /> Platform readiness</a>
        </div>
        <div className="sidebar-group sidebar-bottom">
          <span className="sidebar-kicker">System</span>
          <a href="#settings"><Settings size={16} /> Settings</a>
          <a href="#technology"><Cpu size={16} /> Model information</a>
        </div>
      </aside>
      <nav className="topbar">
        <div className="topbar-context"><span className="status-pip" /> Operations / Overview</div>
        <div className="topbar-tools">
          <form className="global-search" onSubmit={submitSearch} role="search">
            <Search size={15} />
            <input
              value={searchValue}
              onChange={(event) => setSearchValue(event.target.value)}
              placeholder="Search workspace"
              aria-label="Search workspace"
            />
          </form>
          <button className="icon-action" aria-label="Notifications"><Bell size={17} /><span /></button>
          <span className="topbar-date">21 SEP 2026</span>
          <button className="theme-toggle" onClick={onToggleTheme} aria-label="Toggle color theme">
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <button className="primary-btn topbar-launch" onClick={onLaunchRecognition}>Launch scan</button>
        </div>
      </nav>
    </>
  );
}

function SectionLabel({ children }) {
  return <div className="section-label">{children}</div>;
}

function StatCard({ icon: Icon, label, value }) {
  return (
    <div className="stat-card glass-card">
      <div className="stat-icon"><Icon size={18} /></div>
      <div>
        <div className="stat-label">{label}</div>
        <div className="stat-value">{value}</div>
      </div>
    </div>
  );
}

function DashboardSection({ status, onStartRecognition, onViewPipeline }) {
  const statCards = [
    { label: 'Enrolled Members', value: status.people_count, icon: Users },
    { label: 'Model Accuracy', value: '95.74%', icon: Gauge },
    { label: 'Match Threshold', value: '0.45', icon: Layers3 },
  ];

  return (
    <section id="dashboard" className="page-section">
      <div className="hero-grid">
        <div className="hero-copy">
          <SectionLabel>Identity Security Operations</SectionLabel>
          <h1>Secure Every Identity</h1>
          <p>
            A controlled biometric verification console combining face detection and face embeddings for trusted access, rapid investigation, and accountable identity decisions.
          </p>
          <div className="capability-strip" aria-label="Core computer vision capabilities">
            <span>Face Detection</span>
            <span>Face Embeddings</span>
          </div>
          <div className="hero-actions">
            <button className="primary-btn large" onClick={onStartRecognition}>Start Recognition</button>
            <button className="secondary-btn large" onClick={onViewPipeline}>View Pipeline</button>
          </div>
          <div className="metrics-row">
            {statCards.map((card) => (
              <StatCard key={card.label} {...card} />
            ))}
          </div>
        </div>

        <div className="status-panel glass-card">
          <div className="status-topline">
            <span className="mini-dot" />
            <span>System Status</span>
          </div>
          <div className="status-values">
            <div><span>SCRFD</span><strong>{status.scrfd}</strong></div>
            <div><span>ArcFace</span><strong>{status.arcface}</strong></div>
            <div><span>SQLite</span><strong>{status.sqlite}</strong></div>
            <div><span>Recognition Engine</span><strong>{status.recognition_engine}</strong></div>
          </div>
          <div className="signal-core" aria-label="Recognition engine signal visualization">
            <div className="signal-ring signal-ring-one" />
            <div className="signal-ring signal-ring-two" />
            <div className="signal-node">ID</div>
            <span className="signal-caption">EMBEDDING STREAM / NOMINAL</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function RecognitionSection() {
  const [uploadSelectedFile, setUploadSelectedFile] = useState(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState('');
  const [uploadResult, setUploadResult] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  const [webcamActive, setWebcamActive] = useState(false);
  const [webcamError, setWebcamError] = useState('');
  const [webcamCapturedFrameUrl, setWebcamCapturedFrameUrl] = useState('');
  const [webcamResult, setWebcamResult] = useState(null);
  const [webcamLoading, setWebcamLoading] = useState(false);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const captureInProgressRef = useRef(false);
  const refreshTimerRef = useRef(null);

  useEffect(() => () => {
    if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);
  }, []);

  const normalizeSourceValue = (value) => {
    const normalized = String(value || 'upload').trim().toLowerCase();
    return normalized === 'webcam' ? 'Webcam' : 'Upload';
  };

  const buildRecognitionResult = (data, source) => ({
    ...data,
    source: data?.source ?? source,
    status: data?.status || (data?.detected && data?.identity !== 'Unknown' ? 'Recognized' : 'Unknown'),
  });

  const handleUploadFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadSelectedFile(file);
    setUploadPreviewUrl(URL.createObjectURL(file));
    setUploadResult(null);
  };

  const clearUploadSelection = () => {
    setUploadSelectedFile(null);
    setUploadPreviewUrl('');
    setUploadResult(null);
  };

  const submitUploadRecognition = async (fileToUse = uploadSelectedFile) => {
    if (!fileToUse) return;
    setUploadLoading(true);
    setUploadResult(null);

    try {
      const source = 'upload';
      const data = await api.recognize(fileToUse, undefined, source);
      const recognitionResult = buildRecognitionResult(data, source);
      setUploadResult(recognitionResult);
      window.dispatchEvent(new CustomEvent('recognition-recorded'));
      if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = setTimeout(() => {
        window.location.reload();
      }, 8000);
    } catch (error) {
      setUploadResult({
        detected: false,
        identity: 'Unknown',
        similarity: 0,
        threshold: 0.45,
        status: 'Unknown',
        source: 'upload',
        message: getErrorMessage(error, 'Recognition request failed.'),
      });
    } finally {
      setUploadLoading(false);
    }
  };

  const startCamera = async () => {
    setWebcamError('');
    setWebcamCapturedFrameUrl('');
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam API is not supported in this browser.');
      }

      if (streamRef.current) stopCamera();
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error('Webcam preview is unavailable.');

      video.srcObject = stream;
      await video.play();
      if (!video.videoWidth || !video.videoHeight) {
        throw new Error('The webcam started but did not provide a video frame.');
      }
      setWebcamActive(true);
    } catch (error) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setWebcamActive(false);
      setWebcamError(error.message || 'Camera permission denied or unavailable.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setWebcamActive(false);
  };

  const submitWebcamRecognition = async (fileToUse) => {
    if (!fileToUse) return;
    setWebcamLoading(true);
    setWebcamResult(null);

    try {
      const source = 'webcam';
      const data = await api.recognize(fileToUse, undefined, source);
      const recognitionResult = buildRecognitionResult(data, source);
      setWebcamResult(recognitionResult);
      window.dispatchEvent(new CustomEvent('recognition-recorded'));
    } catch (error) {
      setWebcamResult({
        detected: false,
        identity: 'Unknown',
        similarity: 0,
        threshold: 0.45,
        status: 'Unknown',
        source: 'webcam',
        message: getErrorMessage(error, 'Recognition request failed.'),
      });
    } finally {
      setWebcamLoading(false);
    }
    captureInProgressRef.current = false;
  };

  const capturePhoto = async () => {
    if (captureInProgressRef.current || webcamLoading) return;

    if (!videoRef.current || !webcamActive) {
      alert('Start the camera before taking a photo.');
      return;
    }

    const video = videoRef.current;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
      alert('The camera is still starting. Wait for the live image, then take the photo again.');
      return;
    }

    captureInProgressRef.current = true;
    setWebcamCapturedFrameUrl('');
    setWebcamError('');

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      captureInProgressRef.current = false;
      setWebcamError('This browser could not capture a still image from the webcam.');
      return;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const capturedPreview = canvas.toDataURL('image/jpeg', 0.92);
    setWebcamCapturedFrameUrl(capturedPreview);
    stopCamera();

    if (refreshTimerRef.current) clearTimeout(refreshTimerRef.current);
    refreshTimerRef.current = setTimeout(() => {
      window.location.reload();
    }, 8000);

    canvas.toBlob((blob) => {
      if (!blob) {
        captureInProgressRef.current = false;
        alert('Could not capture the webcam image.');
        return;
      }

      const file = new File([blob], 'webcam-capture.png', { type: 'image/png' });
      submitWebcamRecognition(file);
    }, 'image/png');
  };

  const renderRecognitionResult = (result, title) => {
    if (!result) return null;

    return (
      <div className="result-card glass-card">
        <div className="result-pills">
          <span>Face Detection: {result.detected ? 'Detected' : 'Not Detected'}</span>
          <span>Identity: {result.identity || 'Unknown'}</span>
          <span>Similarity: {Number(result.similarity || 0).toFixed(2)}</span>
          <span>Threshold: {Number(result.threshold || 0.45).toFixed(2)}</span>
          <span>Source: {normalizeSourceValue(result.source || title)}</span>
        </div>
        <div className={`identity-line ${result.status === 'Recognized' ? 'success' : 'danger'}`}>
          {result.status === 'Recognized' ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
          <strong>{result.status}</strong>
        </div>
        {result.message && <div className="message-box">{result.message}</div>}
      </div>
    );
  };

  return (
    <section id="recognition" className="page-section">
      <SectionLabel>Face Recognition Console</SectionLabel>
      <div className="recognition-grid">
        <div className="glass-card panel-box">
          <div className="panel-header-row">
            <h3>Upload Image</h3>
          </div>
          <label className="dropzone">
            <input type="file" accept="image/*" hidden onChange={handleUploadFileChange} />
            <Upload size={22} />
            <span>Drag & drop image or browse</span>
          </label>
          {uploadPreviewUrl && (
            <div className="preview-box">
              <img src={uploadPreviewUrl} alt="Upload preview" />
            </div>
          )}
          <div className="camera-actions compact-actions">
            <button className="primary-btn large" onClick={() => submitUploadRecognition()} disabled={!uploadSelectedFile || uploadLoading}>
              {uploadLoading ? 'Processing...' : 'Recognize Face'}
            </button>
            <button className="secondary-btn" onClick={clearUploadSelection} disabled={!uploadSelectedFile && !uploadPreviewUrl}>
              Clear Upload
            </button>
          </div>
          {renderRecognitionResult(uploadResult, 'Upload')}
        </div>

        <div className="glass-card panel-box">
          <div className="panel-header-row">
            <h3>Webcam</h3>
          </div>
          <div className="video-box">
            <video
              ref={videoRef}
              className={webcamCapturedFrameUrl ? 'video-captured' : ''}
              autoPlay
              muted
              playsInline
            />
            {webcamCapturedFrameUrl && <img className="captured-frame" src={webcamCapturedFrameUrl} alt="Captured webcam frame" />}
            <div className="camera-grid" aria-hidden="true" />
            {webcamActive && <div className="frame-scan" aria-hidden="true" />}
            {webcamError && <div className="video-error" role="alert">{webcamError}</div>}
            {!webcamActive && <div className="video-overlay">Camera off</div>}
          </div>
          <div className="camera-actions">
            <button className="primary-btn" onClick={startCamera}>Start Camera</button>
            <button className="secondary-btn" onClick={stopCamera}>Stop Camera</button>
            <button className="secondary-btn" onClick={capturePhoto} disabled={!webcamActive || webcamLoading || captureInProgressRef.current}>Take Photo</button>
          </div>
          {webcamLoading && (
            <div className="result-card glass-card loading-card">
              <CircleDashed className="spin" size={20} />
              <span>Detecting webcam face...</span>
            </div>
          )}
          {renderRecognitionResult(webcamResult, 'Webcam')}
        </div>
      </div>
    </section>
  );
}

function RecognitionHistorySection() {
  const [records, setRecords] = useState([]);
  const [state, setState] = useState('loading');

  useEffect(() => {
    const loadHistory = () => api.getRecognitionHistory()
      .then((data) => {
        setRecords(data.records || []);
        setState('ready');
      })
      .catch(() => setState('error'));

    loadHistory();
    window.addEventListener('recognition-recorded', loadHistory);
    return () => window.removeEventListener('recognition-recorded', loadHistory);
  }, []);

  const formatTime = (value) => {
    if (!value) return '—';
    const parsed = new Date(`${value.replace(' ', 'T')}Z`);
    return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
  };

  const formatSource = (source) => {
    const normalized = String(source || 'upload').trim().toLowerCase();
    return normalized === 'webcam' ? 'Webcam' : 'Upload';
  };

  const getAnalysisLabel = (record) => {
    if (!record) return '—';
    const message = String(record.message || '').trim();
    const source = formatSource(record.source);
    if (record.status === 'Recognized') return source === 'Webcam' ? 'Face matched from webcam' : 'Face matched';
    if (message.toLowerCase().includes('multiple faces')) return 'Multiple faces detected';
    if (message.toLowerCase().includes('no face')) return 'No face detected';
    if (record.status === 'Unknown') return 'Face detected but no matching identity';
    return message || 'Analysis unavailable';
  };

  return (
    <section id="login-details" className="page-section">
      <SectionLabel>Login Details</SectionLabel>
      <div className="people-panel glass-card">
        <div className="panel-header-row">
          <div>
            <h3>Recognized Login Records</h3>
            <p className="panel-subtitle">Every recognition event appears here with the person, date, time, and match result.</p>
          </div>
          {state === 'ready' && <span className="pill">{records.length} records</span>}
        </div>
        {state === 'loading' && <div className="people-state">Loading recognition history...</div>}
        {state === 'error' && (
          <div className="people-state error-state">
            <span>Unable to load recognition history from the server.</span>
            <button className="secondary-btn retry-btn" onClick={() => window.dispatchEvent(new Event('recognition-recorded'))}>Retry</button>
          </div>
        )}
        {state === 'ready' && records.length === 0 && <div className="people-state">No recognition events have been recorded yet.</div>}
        {state === 'ready' && records.length > 0 && (
          <>
            <div className="latest-login-detail">
              <div className="latest-login-heading">
                <span className="status-pip" />
                <span>Latest recognition result</span>
                <time>{formatTime(records[0].created_at)}</time>
              </div>
              <div className="login-detail-grid">
                <div><span>Face Detection</span><strong>{records[0].message === 'No face detected.' ? 'Not Detected' : 'Detected'}</strong></div>
                <div><span>Identity</span><strong>{records[0].person}</strong></div>
                <div><span>Similarity</span><strong>{(Number(records[0].similarity) * 100).toFixed(1)}%</strong></div>
                <div><span>Source</span><strong>{formatSource(records[0].source)}</strong></div>
                <div><span>Analysis</span><strong>{getAnalysisLabel(records[0])}</strong></div>
                <div><span>Status</span><strong className={records[0].status === 'Recognized' ? 'text-success' : 'text-warning'}>{records[0].status === 'Recognized' ? 'Recognized' : records[0].status}</strong></div>
              </div>
              <div className="message-box">{records[0].message || 'Recognition complete.'}</div>
            </div>
            <div className="people-table-wrap">
              <table className="people-table history-table">
                <thead>
                  <tr><th>Person</th><th>Date and time</th><th>Match score</th><th>Source</th><th>Analysis</th><th>Result</th></tr>
                </thead>
                <tbody>
                  {records.map((record) => (
                    <tr key={record.id}>
                      <td><span className="person-avatar">{record.person.slice(0, 1).toUpperCase()}</span>{record.person}</td>
                      <td>{formatTime(record.created_at)}</td>
                      <td>{(Number(record.similarity) * 100).toFixed(1)}%</td>
                      <td>{formatSource(record.source)}</td>
                      <td>{getAnalysisLabel(record)}</td>
                      <td><span className={`status-badge ${record.status === 'Recognized' ? '' : 'status-warning'}`}>{record.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function EnrollmentSection({ onEnrollmentComplete }) {
  const [name, setName] = useState('');
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [resultMessage, setResultMessage] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSelection = (event) => {
    const selected = Array.from(event.target.files || []);
    setFiles(selected);
    setPreviews(selected.map((file) => ({ name: file.name, url: URL.createObjectURL(file) })));
  };

  const submitEnrollment = async () => {
    if (!name.trim() || files.length === 0) return;
    setIsSubmitting(true);
    setResultMessage('');

    try {
      const data = await api.enroll(name, files);
      setResultMessage(data.message || `${data.accepted_count} images processed successfully.`);
      onEnrollmentComplete();
      setName('');
      setFiles([]);
      setPreviews([]);
    } catch (error) {
      setResultMessage(error.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section id="enrollment" className="page-section">
      <SectionLabel>Enrollment</SectionLabel>
      <div className="enrollment-shell glass-card">
        <div className="enrollment-header">
          <h3>Enroll New Person</h3>
          <div className="pill">{files.length} images uploaded</div>
        </div>

        <div className="field-row">
          <label>Person Name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Enter a person name" />
        </div>

        <label className="upload-box">
          <input type="file" multiple accept="image/*" onChange={handleSelection} />
          <FileImage size={18} />
          <span>Select multiple images</span>
        </label>

        <div className="preview-grid">
          {previews.map((preview) => (
            <div key={preview.name} className="thumb-card">
              <img src={preview.url} alt={preview.name} />
              <span>{preview.name}</span>
            </div>
          ))}
        </div>

        <button className="primary-btn large" onClick={submitEnrollment} disabled={isSubmitting || !name || files.length === 0}>
          {isSubmitting ? 'Enrolling...' : 'Enroll Person'}
        </button>

        {resultMessage && <div className="message-box success-box">{resultMessage}</div>}
      </div>
    </section>
  );
}

function PeopleSection({ refreshToken }) {
  const [people, setPeople] = useState([]);
  const [state, setState] = useState('loading');

  useEffect(() => {
    api.getPeople()
      .then((data) => {
        setPeople(data.people || []);
        setState('ready');
      })
      .catch(() => setState('error'));
  }, [refreshToken]);

  return (
    <section id="people" className="page-section">
      <SectionLabel>Enrolled Persons</SectionLabel>
      <div className="people-panel glass-card">
        <div className="panel-header-row">
          <div>
            <h3>People Directory</h3>
            <p className="panel-subtitle">Live records currently stored in the recognition database.</p>
          </div>
          {state === 'ready' && <span className="pill">{people.length} enrolled</span>}
        </div>
        {state === 'loading' && <div className="people-state">Loading enrolled people...</div>}
        {state === 'error' && <div className="people-state error-state">Unable to load people from the recognition server.</div>}
        {state === 'ready' && people.length === 0 && <div className="people-state">No people have been enrolled yet.</div>}
        {state === 'ready' && people.length > 0 && (
          <div className="people-table-wrap">
            <table className="people-table">
              <thead>
                <tr><th>Person</th><th>Face samples</th><th>Status</th></tr>
              </thead>
              <tbody>
                {people.map((person) => (
                  <tr key={person.name}>
                    <td><span className="person-avatar">{person.name.slice(0, 1).toUpperCase()}</span>{person.name}</td>
                    <td>{person.sample_count}</td>
                    <td><span className="status-badge">{person.status || 'Active'}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

function EvaluationSection() {
  const [evaluation, setEvaluation] = useState(null);

  useEffect(() => {
    api.getEvaluation().then(setEvaluation).catch(() => setEvaluation(null));
  }, []);

  return (
    <section id="evaluation" className="page-section">
      <SectionLabel>Model Evaluation Dashboard</SectionLabel>
      <div className="evaluation-grid">
        <div className="glass-card metric-box">
          <div className="metric-title">Known Person Evaluation</div>
          <div className="metric-line"><span>Known Samples</span><strong>{evaluation?.known_samples ?? 47}</strong></div>
          <div className="metric-line"><span>Correct Identifications</span><strong>{evaluation?.correct_identifications ?? 45}</strong></div>
          <div className="metric-line"><span>Accuracy</span><strong>{evaluation ? `${(evaluation.accuracy * 100).toFixed(2)}%` : '95.74%'}</strong></div>
          <div className="metric-line"><span>Rejected as Unknown</span><strong>{evaluation?.rejected_as_unknown ?? 2}</strong></div>
          <div className="metric-line"><span>Threshold</span><strong>{evaluation?.threshold ?? 0.45}</strong></div>
        </div>

        <div className="glass-card metric-box">
          <div className="metric-title">Unknown Person Evaluation</div>
          <div className="metric-line"><span>Unknown Samples</span><strong>{evaluation?.unknown_samples ?? '—'}</strong></div>
          <div className="metric-line"><span>Correctly Rejected</span><strong>{evaluation?.correctly_rejected_unknown_samples ?? '—'}</strong></div>
          <div className="metric-line"><span>Unknown Rejection Rate</span><strong>{evaluation ? `${(evaluation.unknown_rejection_rate * 100).toFixed(2)}%` : '—'}</strong></div>
          <div className="metric-line"><span>Threshold</span><strong>{evaluation?.threshold ?? 0.45}</strong></div>
        </div>
      </div>
    </section>
  );
}

function PipelineSection() {
  const steps = [
    'Input Image / Webcam',
    'SCRFD',
    'Face Detection',
    'ArcFace',
    'Face Embedding',
    'SQLite',
    'Stored Embeddings',
    'Cosine Similarity',
    'Matching',
    'Threshold',
    'Recognized / Unknown',
  ];

  return (
    <section id="how-it-works" className="page-section">
      <SectionLabel>How It Works</SectionLabel>
      <div className="pipeline">
        {steps.map((step, index) => (
          <div key={step} className="pipeline-step">
            <span>{step}</span>
            {index < steps.length - 1 && <ArrowRight size={16} />}
          </div>
        ))}
      </div>
    </section>
  );
}

function TechnologySection() {
  const tech = [
    ['Python + FastAPI', 'Backend API, recognition endpoints, enrollment, history, and health checks'],
    ['SCRFD', 'Detects faces and returns face regions and landmarks from uploaded or webcam frames'],
    ['ArcFace', 'Converts each detected face into a normalized identity embedding'],
    ['buffalo_l', 'InsightFace model pack containing the detector and recognition models used by the engine'],
    ['OpenCV', 'Decodes uploaded files, captures webcam frames, and prepares images for inference'],
    ['ONNX Runtime', 'Runs the SCRFD and ArcFace ONNX models locally on the CPU'],
    ['SQLite', 'Stores enrolled embeddings, recognition source, score, threshold, status, and event time'],
    ['Cosine Similarity', 'Compares the live embedding with enrolled embeddings to find the closest identity'],
    ['React + Vite', 'Provides the responsive security dashboard and separate upload/webcam workflows'],
  ];

  return (
    <section id="technology" className="page-section">
      <SectionLabel>Model Information</SectionLabel>
      <div className="technology-intro">
        <h2>FaceSecure AI recognition stack</h2>
        <p>
          This project is a local face recognition security console. An image from the Upload or Webcam workflow is decoded by OpenCV, scanned by SCRFD, converted into an ArcFace embedding, and compared with enrolled embeddings. The match score is evaluated against the configured threshold, then the result and its real source are recorded in SQLite.
        </p>
      </div>
      <div className="technology-grid">
        {tech.map(([name, description]) => (
          <div key={name} className="glass-card tech-card">
            <div className="tech-name">{name}</div>
            <p>{description}</p>
          </div>
        ))}
      </div>
      <div className="model-flow glass-card">
        <div><span>Input</span><strong>Uploaded image or one webcam frame</strong></div>
        <div><span>Detection</span><strong>SCRFD finds exactly one usable face</strong></div>
        <div><span>Embedding</span><strong>ArcFace creates the identity vector</strong></div>
        <div><span>Decision</span><strong>Cosine score is checked against threshold</strong></div>
        <div><span>Audit</span><strong>SQLite stores the result and source</strong></div>
      </div>
    </section>
  );
}

function AboutSection() {
  const failureCases = [
    'Dark images',
    'Blurry images',
    'Low-resolution images',
    'Extreme side angles',
    'Masks',
    'Sunglasses',
    'Heavy occlusion',
    'Multiple faces',
    'No face detected',
    'Similar-looking people',
    'Incorrect threshold',
  ];

  return (
    <section id="about" className="page-section">
      <SectionLabel>Security & Privacy</SectionLabel>
      <div className="about-grid">
        <div className="glass-card info-box">
          <h3>Biometric Data Protection</h3>
          <ul>
            <li>User consent and transparent handling</li>
            <li>Secure database access</li>
            <li>Access control and deletion workflows</li>
            <li>Responsible biometric-data handling</li>
            <li>Encryption as a future improvement if implemented</li>
          </ul>
        </div>

        <div className="glass-card info-box">
          <h3>Failure Cases</h3>
          <div className="chips-grid">
            {failureCases.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function PlatformReadinessSection() {
  const areas = [
    ['Recognition controls', 'Upload and webcam recognition events are recorded with their actual source and match result.'],
    ['Data integrity', 'Recognition history is stored in SQLite with source, similarity, threshold, and event time.'],
    ['Settings', 'Recognition threshold is already API-backed; camera and notification settings are ready for extension.'],
    ['Model information', 'SCRFD, ArcFace, embeddings, SQLite, and ONNX Runtime are active in the current engine.'],
  ];

  return (
    <section id="platform-readiness" className="page-section readiness-section">
      <SectionLabel>Platform Readiness</SectionLabel>
      <div className="readiness-grid">
        {areas.map(([title, description]) => (
            <article className="glass-card readiness-card" id={title === 'Model information' ? 'model' : title.toLowerCase().replaceAll(' ', '-')} key={title}>
            <span className="readiness-status">API READY</span>
            <h3>{title}</h3>
            <p>{description}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

export default function App() {
  const [status, setStatus] = useState({
    scrfd: 'Loading',
    arcface: 'Loading',
    sqlite: 'Loading',
    recognition_engine: 'Loading',
    people_count: 0,
    roster_count: 0,
  });
  const [theme, setTheme] = useState('dark');
  const [peopleRefreshToken, setPeopleRefreshToken] = useState(0);

  const workspaceTargets = {
    dashboard: 'dashboard',
    people: 'people',
    recognition: 'recognition',
    webcam: 'recognition',
    upload: 'recognition',
    login: 'login-details',
    history: 'login-details',
    enrollment: 'enrollment',
    pipeline: 'how-it-works',
    technology: 'technology',
    model: 'technology',
    about: 'about',
    platform: 'platform-readiness',
    readiness: 'platform-readiness',
    settings: 'platform-readiness',
  };

  const refreshPeopleCount = () => {
    api.getStatus()
      .then((data) => setStatus({
        scrfd: data.scrfd,
        arcface: data.arcface,
        sqlite: data.sqlite,
        recognition_engine: data.recognition_engine,
        people_count: data.people_count ?? 0,
        roster_count: data.roster_count ?? 0,
      }))
      .catch(() => setStatus({
        scrfd: 'Unavailable',
        arcface: 'Unavailable',
        sqlite: 'Unavailable',
        recognition_engine: 'Unavailable',
        people_count: 0,
        roster_count: 0,
      }));
    setPeopleRefreshToken((token) => token + 1);
  };

  useEffect(() => {
    refreshPeopleCount();
  }, []);

  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const searchWorkspace = (query) => {
    const normalizedQuery = query.trim().toLowerCase();
    const matchingKey = Object.keys(workspaceTargets).find((key) => normalizedQuery.includes(key));
    if (matchingKey) scrollToSection(workspaceTargets[matchingKey]);
  };

  return (
    <div className={`app-shell theme-${theme}`}>
      <div className="cinematic-loader" aria-hidden="true">
        <div className="loader-grid" />
        <div className="loader-content">
          <div className="loader-mark">F</div>
          <div className="loader-kicker">Identity security operations</div>
          <strong>FaceSecure AI</strong>
          <div className="loader-progress"><span /></div>
          <div className="loader-status">Initializing recognition console</div>
        </div>
      </div>
      <Navbar
        theme={theme}
        onToggleTheme={() => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))}
        onLaunchRecognition={() => scrollToSection('recognition')}
        onWorkspaceSearch={searchWorkspace}
      />
      <main className="main-content">
        <DashboardSection
          status={status}
          onStartRecognition={() => scrollToSection('recognition')}
          onViewPipeline={() => scrollToSection('how-it-works')}
        />
        <RecognitionSection />
        <RecognitionHistorySection />
        <EnrollmentSection onEnrollmentComplete={refreshPeopleCount} />
        <PeopleSection refreshToken={peopleRefreshToken} />
        <EvaluationSection />
        <PipelineSection />
        <TechnologySection />
        <AboutSection />
        <PlatformReadinessSection />
      </main>
    </div>
  );
}
