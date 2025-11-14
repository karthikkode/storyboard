// app/page.tsx
'use client';
export const runtime = 'nodejs';
import { useState } from 'react';
// We will create this file next
import { generateStoryboard } from './actions'; 

// Define the structure of our final scene, based on your sample JSON
interface Scene {
  scene: number;
  start_time_secs: number;
  end_time_secs: number;
  script: string;
  prompt: string;
  image_url: string;
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [scenes, setScenes] = useState<Scene[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null); // Clear previous errors
      setScenes([]); // Clear previous results
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an audio file first.');
      return;
    }

    setIsLoading(true);
    setStatusMessage('Starting process...');
    setError(null);
    setScenes([]);

    try {
      // Use FormData to send the file to the Server Action
      const formData = new FormData();
      formData.append('audioFile', file);

      // This is where the magic happens. We call our server function
      // directly from the client.
      const result = await generateStoryboard(formData);

      if (result.error) {
        throw new Error(result.error);
      }
      
      setScenes(result.scenes || []);
      setStatusMessage('Storyboard generation complete!');

    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
      setStatusMessage('');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="container mx-auto p-8">
      <h1 className="text-4xl font-bold mb-6 text-center">
        Storyboard Generator 🎬
      </h1>

      <form onSubmit={handleSubmit} className="max-w-xl mx-auto space-y-4">
        <div>
          <label
            htmlFor="audioFile"
            className="block text-sm font-medium text-gray-700"
          >
            Upload your audio file (WAV, MP3, FLAC):
          </label>
          <input
            id="audioFile"
            name="audioFile"
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            disabled={isLoading}
            className="mt-1 block w-full text-sm text-gray-500
                       file:mr-4 file:py-2 file:px-4
                       file:rounded-full file:border-0
                       file:text-sm file:font-semibold
                       file:bg-violet-50 file:text-violet-700
                       hover:file:bg-violet-100"
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full px-6 py-3 border border-transparent text-base font-medium rounded-md text-white 
                     bg-indigo-600 shadow-sm hover:bg-indigo-700 
                     disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Generating...' : 'Generate Storyboard'}
        </button>
      </form>

      {/* --- STATUS & RESULTS --- */}
      <div className="max-w-3xl mx-auto mt-10">
        {isLoading && (
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <p className="font-semibold text-blue-700">Working on it...</p>
            <p className="text-blue-600">{statusMessage}</p>
          </div>
        )}

        {error && (
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <p className="font-semibold text-red-700">Error</p>
            <p className="text-red-600">{error}</p>
          </div>
        )}

        {scenes.length > 0 && (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-center">Your Storyboard</h2>
            {scenes.map((scene) => (
              <div
                key={scene.scene}
                className="flex flex-col md:flex-row gap-4 p-4 border rounded-lg shadow-md bg-white"
              >
                <div className="flex-1">
                  <h3 className="text-xl font-semibold">Scene {scene.scene}</h3>
                  <p className="text-sm text-gray-500">
                    ({scene.start_time_secs.toFixed(1)}s - {scene.end_time_secs.toFixed(1)}s)
                  </p>
                  <p className="mt-2 text-gray-700">{scene.script}</p>
                  <p className="mt-2 text-xs text-gray-400 italic">
                    <span className="font-medium">Prompt:</span> {scene.prompt}
                  </p>
                </div>
                <div className="md:w-1/3">
                  {scene.image_url.startsWith('http') ? (
                    <img
                      src={scene.image_url}
                      alt={`Storyboard image for scene ${scene.scene}`}
                      className="w-full h-auto rounded-lg object-cover aspect-video"
                    />
                  ) : (
                    <div className="w-full aspect-video bg-gray-200 rounded-lg flex items-center justify-center">
                      <p className="text-red-500 text-xs text-center p-2">{scene.image_url}</p>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}