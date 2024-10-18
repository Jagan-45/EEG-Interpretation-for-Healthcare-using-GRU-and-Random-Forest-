import React, { useState } from 'react';
import { Activity, Brain, FileUp, Heart, Home, Menu, User, AlertTriangle, Mail, Loader, XCircle } from 'lucide-react';

const LoaderComponent = () => (
  <div className="flex items-center justify-center h-full">
    <Loader className="animate-spin h-10 w-10 text-blue-500" />
    <span className="ml-2 text-blue-500">Loading...</span>
  </div>
);

const ErrorComponent = ({ message }) => (
  <div className="flex items-center justify-center p-4 bg-red-100 border border-red-400 text-red-700 rounded relative">
    <XCircle className="h-6 w-6 mr-2" />
    <strong>Error:</strong> {message}
  </div>
);

export default function EEGHealthApp() {
  const [activeTab, setActiveTab] = useState('home');
  const [emotionFile, setEmotionFile] = useState(null);
  const [activityFile, setActivityFile] = useState(null);
  const [epilepsyFile, setEpilepsyFile] = useState(null);
  const [emotionalState, setEmotionalState] = useState(null);
  const [detectedActivity, setDetectedActivity] = useState(null);
  const [epilepsyDiagnosis, setEpilepsyDiagnosis] = useState(null);
  const [plotImage, setPlotImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleEmotionFileUpload = (event) => {
    const file = event.target.files[0];
    setEmotionFile(file);
    setLoading(true);
    setError(null); // Reset error state

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:5000/api/emotion', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        setLoading(false);
        if (data.error) {
          setEmotionalState(null);
          setError(data.error);
          console.error('Error:', data.error);
        } else {
          setEmotionalState(data.emotion);
          setPlotImage(`http://localhost:5000/${data.plot}`);
        }
      })
      .catch((error) => {
        setLoading(false);
        setError('Failed to fetch emotion data.');
        console.error('Error:', error);
      });
  };

  const handleActivityFileUpload = (event) => {
    const file = event.target.files[0];
    setActivityFile(file);
    setLoading(true);
    setError(null); // Reset error state

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:5000/api/activity', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        setLoading(false);
        if (data.error) {
          setDetectedActivity(null);
          setError(data.error);
          console.error('Error:', data.error);
        } else {
          const activity_mapping = {
            "0": "Working on a desk",
            "1": "Sitting idle",
            "2": "Laying down",
            "3": "Sitting",
            "4": "Standing",
            "5": "Watching TV",
            "6": "Cooking",
            "7": "Stairs",
            "8": "Walking",
            "9": "Jogging",
            "10": "Running",
            "11": "Jumping"
          };
          setDetectedActivity(activity_mapping[data.activity]);
        }
      })
      .catch((error) => {
        setLoading(false);
        setError('Failed to fetch activity data.');
        console.error('Error:', error);
      });
  };

  const handleEpilepsyFileUpload = (event) => {
    const file = event.target.files[0];
    setEpilepsyFile(file);
    setLoading(true);
    setError(null); // Reset error state

    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:5000/api/activity', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        setLoading(false);
        if ( data.error) {
          setEpilepsyDiagnosis(null);
          setError(data.error);
          console.error('Error:', data.error);
        } else {
          setEpilepsyDiagnosis(data.epileptic ? 'Yes' : 'No');
        }
      })
      .catch((error) => {
        setLoading(false);
        setError('Failed to fetch epilepsy data.');
        console.error('Error:', error);
      });
  };


  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-64 bg-blue-800 text-white">
        <div className="p-4">
          <h1 className="text-2xl font-bold flex items-center">
            <Brain className="mr-2" /> EEG Health
          </h1>
        </div>
        <nav className="mt-8">
          <button onClick={() => setActiveTab('home')} className={`block w-full text-left py-2 px-4 ${activeTab === 'home' ? 'bg-blue-900' : 'hover:bg-blue-700'}`}>
            <Home className="inline-block mr-2" /> Home
          </button>
          <button onClick={() => setActiveTab('emotion')} className={`block w-full text-left py-2 px-4 ${activeTab === 'emotion' ? 'bg-blue-900' : 'hover:bg-blue-700'}`}>
            <Heart className="inline-block mr-2" /> Emotion Analysis
          </button>
          <button onClick={() => setActiveTab('activity')} className={`block w-full text-left py-2 px-4 ${activeTab === 'activity' ? 'bg-blue-900' : 'hover:bg-blue-700'}`}>
            <Activity className="inline-block mr-2" /> Activity Tracking
          </button>
          <button onClick={() => setActiveTab('epilepsy')} className={`block w-full text-left py-2 px-4 ${activeTab === 'epilepsy' ? 'bg-blue-900' : 'hover:bg-blue-700'}`}>
            <AlertTriangle className="inline-block mr-2" /> Epilepsy Diagnosis
          </button>
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="flex items-center justify-between p-4">
            <div className="flex items-center">
              <button className="mr-4">
                <Menu className="h-6 w-6 text-gray-500" />
              </button>
              <h2 className="text-xl font-semibold text-gray-800">
                {activeTab === 'home' ? 'Welcome to EEG Health' : 
                 activeTab === 'emotion' ? 'Emotion Analysis' : 
                 activeTab === 'activity' ? 'Activity Tracking' : 'Epilepsy Diagnosis'}
              </h2>
            </div>
            <div className="flex items-center">
              <button className="p-2 rounded-full hover:bg-gray-200">
                <User className="h-6 w-6 text-gray-500" />
              </button>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 p-6 bg-white overflow-y-auto">
          {activeTab === 'home' && (
            <div className="text-center">
              <Brain className="w-24 h-24 mx-auto mb-6 text-blue-600" />
              <h2 className="text-3xl font-bold mb-4">Welcome to EEG Health</h2>
              <p className="text-xl mb-8">Analyze emotions, track activities, and diagnose epilepsy with advanced EEG technology</p>
              <div className="grid grid-cols-3 gap-8 max-w-4xl mx-auto mb-12">
                <div className="border p-6 rounded-lg shadow-md">
                  <Heart className="w-12 h-12 mx-auto mb-4 text-red-500" />
                  <h3 className="text-xl font-semibold mb-2">Emotion Analysis</h3>
                  <p>Upload EEG data to analyze emotional states and view detailed graphs.</p>
                </div>
                <div className="border p-6 rounded-lg shadow-md">
                  <Activity className="w-12 h-12 mx-auto mb-4 text-green-500" />
                  <h3 className="text-xl font-semibold mb-2">Activity Tracking</h3>
                  <p>Monitor patient activities with AI-powered avatars and real-time updates.</p>
                </div>
                <div className="border p-6 rounded-lg shadow-md">
                  <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-yellow-500" />
                  <h3 className="text-xl font-semibold mb-2">Epilepsy Diagnosis</h3>
                  <p>Analyze EEG data to diagnose epilepsy with high accuracy.</p>
                </div>
              </div>
              <div className="max-w-2xl mx-auto bg-gray-100 p-8 rounded-lg">
                <h3 className="text-2xl font-bold mb-4">Contact Us</h3>
                <p className="mb-4">Have questions or need support? Reach out to our team.</p>
                <div className="flex items-center justify-center space-x-4">
                  <Mail className="w-6 h-6 text-blue-600" />
                  <a href="mailto:support@eeghealth.com" className="text-blue-600 hover:underline">support@eeghealth.com</a>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'emotion' && (
            <div>
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload EEG CSV File for Emotion Analysis
                </label>
                <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                  <div className="space-y-1 text-center">
                    <FileUp className="mx-auto h-12 w-12 text-gray-400" />
                    <div className="flex text-sm text-gray-600">
                      <label htmlFor="emotion-file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500">
                        <span>Upload a file</span>
                        <input id="emotion-file-upload" name="emotion-file-upload" type="file" className="sr-only" onChange={handleEmotionFileUpload} />
                      </label>
                      <p className="pl-1">or drag and drop</p>
                    </div>
                    <p className="text-xs text-gray-500">CSV file up to 10MB</p>
                  </div>
                </div>
              </div>
              {loading ? (
 <LoaderComponent />
              ) : error ? (
                <ErrorComponent message={error} />
              ) : emotionalState && plotImage ? (
                <div>
                  <h3 className="text-xl font-semibold mb-2">Emotion Analysis Results</h3>
                  <p className="text-lg mb-4">Emotional State: <span className="font-semibold text-blue-600">{emotionalState}</span></p>
                  <img src={plotImage} alt="EEG Signal Plot" className="w-full h-64 object-cover rounded-lg mb-4" />
                </div>
              ) : null}
            </div>
          )}
            

            {activeTab === 'activity' && (
                <div>
                  <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Upload CSV File for Activity Tracking
                    </label>
                    <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                      <div className="space-y-1 text-center">
                        <FileUp className="mx-auto h-12 w-12 text-gray-400" />
                        <div className="flex text-sm text-gray-600">
                          <label htmlFor="activity-file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500">
                            <span>Upload a file</span>
                            <input id="activity-file-upload" name="activity-file-upload" type="file" className="sr-only" onChange={handleActivityFileUpload} />
                          </label>
                          <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className="text-xs text-gray-500">CSV file up to 10MB</p>
                      </div>
                    </div>
                  </div>
                  {loading ? (
                    <LoaderComponent />
                  ) : error ? (
                    <ErrorComponent message={error} />
                  ) : detectedActivity ? (
                    <div>
                      <h3 className="text-xl font-semibold mb-2">Activity Tracking Results</h3>
                      <p className="text-lg mb-4">Detected Activity: <span className="font-semibold text-green-600">{detectedActivity}</span></p>
                      
                      {/* GIF Display */}
                      <div className="flex justify-center mb-4">
                        <img 
                          src={`/home/jagan/Development/eeg_final/frontend/src/assets/${detectedActivity}.webp`} 
                          alt={detectedActivity} 
                          className="w-full max-w-xs h-auto rounded-lg shadow-lg" 
                          type="image/webp"
                        />
                      </div>
                    </div>
                  ) : null}
                </div>
              )}
          {activeTab === 'epilepsy' && (
            <div>
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload EEG CSV File for Epilepsy Diagnosis
                </label>
                <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                  <div className="space-y-1 text-center">
                    <FileUp className="mx-auto h-12 w-12 text-gray-400" />
                    <div className="flex text-sm text-gray-600">
                      <label htmlFor="epilepsy-file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500">
                        <span>Upload a file</span>
                        <input id="epilepsy-file-upload" name="epilepsy-file-upload" type="file" className="sr-only" onChange={handleEpilepsyFileUpload} />
                      </label>
                      <p className="pl-1">or drag and drop</p>
                    </div>
                    <p className="text-xs text-gray-500">CSV file up to 10MB</p>
                  </div>
                </div>
              </div>
              {loading ? (
                <LoaderComponent />
              ) : error ? (
                <ErrorComponent message={error} />
              ) : epilepsyDiagnosis ? (
                <div>
                  <h3 className="text-xl font-semibold mb-2">Epilepsy Diagnosis Results</h3>
                  <p className="text-lg mb-4">
                    Diagnosis: 
                    <span className={`font-semibold ${epilepsyDiagnosis === 'Yes' ? 'text-red-600' : 'text-green-600'}`}>
                      {epilepsyDiagnosis}
                    </span>
                  </p>
                  <div className="bg-gray-100 p-4 rounded-lg">
                    <p className="text-sm text-gray-600">
                      {epilepsyDiagnosis === 'Yes' 
                        ? 'The EEG data suggests a positive diagnosis for epilepsy. Please consult with a healthcare professional for further evaluation and treatment options.'
                        : 'The EEG data does not indicate epilepsy. However, please consult with a healthcare professional for a comprehensive evaluation.'}
                    </p>
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
