// app/actions.ts
'use server';

import path from 'path';
import { URL } from 'url';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegStatic from 'ffmpeg-static';
import fs from 'fs/promises';
import { constants as FS } from 'fs';

import { Storage } from '@google-cloud/storage';
import { SpeechClient } from '@google-cloud/speech';
import { GoogleGenAI, HarmCategory, HarmBlockThreshold } from '@google/genai';
import { protos } from '@google-cloud/speech';

// --- FFmpeg path resolution (robust on Windows) ---
const resolveFfmpegPath = () => {
  const p = (ffmpegStatic as unknown as string) || process.env.FFMPEG_PATH || '';
  if (!p) return '';
  const abs = path.isAbsolute(p) ? p : path.resolve(p);
  return abs.replace('app.asar', 'app.asar.unpacked');
};

const ffmpegBin = resolveFfmpegPath();
if (!ffmpegBin) {
  console.error('FFmpeg binary path not found. Set FFMPEG_PATH or install ffmpeg-static.');
} else {
  console.log('Using FFmpeg binary:', ffmpegBin);
  ffmpeg.setFfmpegPath(ffmpegBin);
}

// Ensure FFmpeg exists before any job runs
async function assertFfmpeg(): Promise<void> {
  if (!ffmpegBin) throw new Error('FFmpeg path not resolved');
  await fs.access(ffmpegBin, FS.F_OK).catch(() => {
    throw new Error(`FFmpeg not found at: ${ffmpegBin}`);
  });
}

// --- 1. INITIALIZE CLIENTS AND CONSTANTS ---

const storage = new Storage({
  projectId: process.env.GOOGLE_PROJECT_ID,
});
const speechClient = new SpeechClient({
  projectId: process.env.GOOGLE_PROJECT_ID,
});

const googleAI = new GoogleGenAI({
  vertexai: true,
  project: process.env.GOOGLE_PROJECT_ID!,
  location: process.env.GOOGLE_LOCATION!,
});

// Buckets
const audioBucketName = process.env.AUDIO_BUCKET_NAME!;
const transcriptBucketName = process.env.TRANSCRIPT_BUCKET_NAME!;
const chunkedBucketName = process.env.CHUNKED_BUCKET_NAME!;
const imageBucketName = process.env.IMAGE_BUCKET_NAME!;
const videoBucketName = process.env.VIDEO_BUCKET_NAME!;

// Model Names
const TEXT_MODEL = 'gemini-2.5-pro';
const IMAGE_MODEL = 'imagen-4.0-ultra-generate-001';
const DEFAULT_AR = '16:9';
const SLEEP_BETWEEN_IMAGES_SEC = 30;

// --- 2. HELPER FUNCTIONS ---

function getJobId(filename: string): string {
  const fileExtension = filename.lastIndexOf('.');
  const name = fileExtension === -1 ? filename : filename.substring(0, fileExtension);
  return name.toLowerCase().replace(/[^a-z0-9]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

async function saveTxtTranscript(jobId: string, script: string): Promise<void> {
  const filePath = `${jobId}.txt`;
  try {
    console.log(`SAVING transcript: ${filePath} in bucket ${transcriptBucketName}`);
    const file = storage.bucket(transcriptBucketName).file(filePath);
    await file.save(script, { contentType: 'text/plain; charset=utf-8' });
    console.log(`SAVED transcript: ${filePath}`);
  } catch (error) {
    console.error(`Error saving .txt transcript ${filePath}:`, error);
  }
}

type Scene = {
  scene: number;
  start_time_secs: number;
  end_time_secs: number;
  script: string;
  prompt: string | null;
  image_url: string | null;
};

async function saveChunkedJson(jobId: string, scenes: Scene[]): Promise<void> {
  const filePath = `${jobId}.json`;
  try {
    console.log(`SAVING chunked JSON: ${filePath} in bucket ${chunkedBucketName}`);
    const file = storage.bucket(chunkedBucketName).file(filePath);
    await file.save(JSON.stringify(scenes, null, 2), { contentType: 'application/json; charset=utf-8' });
    console.log(`SAVED chunked JSON: ${filePath}`);
  } catch (error) {
    console.error(`Error saving chunked JSON ${filePath}:`, error);
  }
}

async function downloadGCSFile(bucketName: string, fileName: string, localPathStr: string): Promise<void> {
  console.log(`Downloading gs://${bucketName}/${fileName} to ${localPathStr}...`);
  const options = { destination: localPathStr };
  await storage.bucket(bucketName).file(fileName).download(options);
  console.log('Download complete.');
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// --- 3. TYPES ---
type WordInfo = { word: string; startTime: number; endTime: number };

// --- 3.1 JSON SANITIZER FOR MODEL OUTPUT ---
function extractPossibleJSON(text: string): string {
  if (!text) return text;
  const trimmed = text.trim();
  const fenceMatch = trimmed.match(/```(?:json)?\n([\s\S]*?)```/i);
  if (fenceMatch && fenceMatch[1]) return fenceMatch[1].trim();
  return trimmed;
}

function removeTrailingCommas(jsonLike: string): string {
  return jsonLike
    .replace(/,\s*([}\]])/g, '$1')
    .replace(/\u201c|\u201d|\u2018|\u2019/g, '"');
}

function coerceSceneId(sceneId: unknown): number | null {
  if (typeof sceneId === 'number' && Number.isFinite(sceneId)) return sceneId;
  if (typeof sceneId === 'string') {
    const num = parseInt(sceneId.replace(/[^0-9]/g, ''), 10);
    return Number.isFinite(num) ? num : null;
  }
  return null;
}

function extractSceneId(obj: any): number | null {
  if (!obj || typeof obj !== 'object') return null;
  const candidate =
    (obj as any).scene_id ??
    (obj as any).scene ??
    (obj as any).id ??
    (obj as any).index ??
    null;
  return coerceSceneId(candidate);
}

function parseModelJSON(raw: string): any {
  try {
    return JSON.parse(raw);
  } catch {
    const cleaned = removeTrailingCommas(extractPossibleJSON(raw));
    return JSON.parse(cleaned);
  }
}

// --- 4. API WORKFLOW STEPS ---

// STEP 1: TRANSCRIBE
async function runStep1_Transcribe(
  fullJobId: string,
  file: File
): Promise<{ words: WordInfo[]; gcsAudioUri: string }> {
  console.log(`RUNNING: Step 1 Transcription for job ${fullJobId}`);
  const fileExtension = file.name.substring(file.name.lastIndexOf('.'));
  const audioFileName = `${fullJobId}${fileExtension}`;
  const audioFilePath = `gs://${audioBucketName}/${audioFileName}`;
  console.log(`Uploading audio to: ${audioFilePath}`);

  const audioBlob = storage.bucket(audioBucketName).file(audioFileName);
  const stream = audioBlob.createWriteStream({ resumable: false, contentType: file.type });
  const fileBuffer = await file.arrayBuffer();
  await new Promise((resolve, reject) => {
    stream.on('error', reject);
    stream.on('finish', resolve);
    stream.end(Buffer.from(fileBuffer));
  });

  console.log('Audio upload complete.');

  console.log('Starting Speech-to-Text...');
  const config: protos.google.cloud.speech.v1.IRecognitionConfig = {
    encoding: 'LINEAR16',
    sampleRateHertz: 48000,
    languageCode: 'te-IN',
    enableWordTimeOffsets: true,
    enableAutomaticPunctuation: true,
    model: 'latest_long',
    audioChannelCount: 1,
    useEnhanced: true,
  };
  const audio: protos.google.cloud.speech.v1.IRecognitionAudio = { uri: audioFilePath };

  const [operation] = await speechClient.longRunningRecognize({ config, audio });
  console.log('Waiting for transcription operation to complete...');
  const [response] = await operation.promise({ timeout: 3600000 });
  console.log('Speech-to-Text complete.');

  const words: WordInfo[] = [];
  response.results?.forEach(result => {
    result.alternatives?.[0].words?.forEach(wordInfo => {
      if (wordInfo.word && wordInfo.startTime && wordInfo.endTime) {
        words.push({
          word: wordInfo.word,
          startTime:
            parseFloat(wordInfo.startTime.seconds || '0') + (wordInfo.startTime.nanos || 0) / 1e9,
          endTime:
            parseFloat(wordInfo.endTime.seconds || '0') + (wordInfo.endTime.nanos || 0) / 1e9,
        });
      }
    });
  });

  if (words.length === 0) throw new Error('Transcription produced zero words.');

  const fullScriptWithTimestamps = words
    .map(w => `[${w.startTime.toFixed(1)}s - ${w.endTime.toFixed(1)}s] ${w.word}`)
    .join('\n');
  await saveTxtTranscript(fullJobId, fullScriptWithTimestamps);

  return { words, gcsAudioUri: audioFilePath };
}

// STEP 2: CHUNK SCRIPT
async function runStep2_ChunkScript(jobId: string, words: WordInfo[]): Promise<Scene[]> {
  console.log('RUNNING: Step 2 Chunking script...');
  const SENTENCE_END_RE = /[.?!]|।/;
  const TARGET_DURATION_SEC = 7.0;

  const scenes: Scene[] = [];
  if (words.length === 0) return scenes;

  let currentTokens: string[] = [];
  let currentStart = words[0].startTime;

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const token = w.word || '';
    currentTokens.push(token);
    const isLast = i === words.length - 1;
    const curDuration = w.endTime - currentStart;
    let endHere = false;

    if (SENTENCE_END_RE.test(token)) {
      if (curDuration >= TARGET_DURATION_SEC) endHere = true;
    }
    if (isLast) endHere = true;

    if (endHere && currentTokens.length > 0) {
      scenes.push({
        scene: scenes.length + 1,
        start_time_secs: currentStart,
        end_time_secs: w.endTime,
        script: currentTokens.join(' ').trim(),
        prompt: null,
        image_url: null,
      });
      currentTokens = [];
      if (!isLast) currentStart = words[i + 1].startTime;
    }
  }

  await saveChunkedJson(jobId, scenes);
  console.log(`Chunking complete. Created ${scenes.length} scenes.`);
  return scenes;
}

// STEP 3: ANALYZE & GENERATE PROMPTS
async function runStep3_AnalyzeAndGeneratePrompts(
  jobId: string,
  scenes: Scene[],
  fullScript: string
): Promise<Scene[]> {
  console.log(`RUNNING: Step 3 Analyze and Generate Prompts for job ${jobId}...`);

  const sceneList = scenes.map(s => `Scene ${s.scene}: "${s.script}"`).join('\n');

  const prompt = `You are an expert storyboard prompt generator.

FIRST, analyze this FULL SCRIPT to determine its overall 'backdrop' and 'tone':
--- FULL SCRIPT ---
${fullScript}
--- END FULL SCRIPT ---

SECOND, using that analysis, generate one unique image prompt for EACH of the following scenes.

--- RULES ---
1) The prompt MUST follow this exact format: "A cartoon, anime or Ghibli kind of art in [backdrop] and in [tone] that [visual description of scene script]. 16:9."
2) Replace [backdrop] and [tone] with your analysis.
3) **CRITICAL SAFETY RULE: For any sensitive or violent content, you MUST generate an abstract, symbolic, or non-literal visual description that avoids direct depiction of violence, blood, or explicit harm. Focus on emotion, consequences, or atmosphere (e.g., 'a somber landscape' instead of 'a killing scene').**
4) Replace [visual description of scene script] with a concise visual summary of the scene's script, ensuring it fully complies with Rule 3.
5) Return ONLY minified JSON. Do not wrap in code fences. Do not add any commentary.

--- SCENE LIST ---
${sceneList}
--- END SCENE LIST ---

Return ONLY the requested JSON with 'analysis' and 'scenes'.`;

  const schema = {
    type: 'OBJECT',
    properties: {
      analysis: {
        type: 'OBJECT',
        properties: {
          backdrop: { type: 'STRING' },
          tone: { type: 'STRING' },
        },
        required: ['backdrop', 'tone'],
      },
      scenes: {
        type: 'ARRAY',
        items: {
          type: 'OBJECT',
          properties: {
            scene_id: { type: 'STRING' },
            scene: { type: 'STRING' },
            id: { type: 'STRING' },
            index: { type: 'STRING' },
            prompt: { type: 'STRING' },
          },
          required: ['prompt'],
          additionalProperties: true,
        },
      },
    },
    required: ['analysis', 'scenes'],
    additionalProperties: true,
  } as const;

  console.log('Calling Gemini for prompt generation...');
  const result = await googleAI.models.generateContent({
    model: TEXT_MODEL,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: schema as any,
    },
    safetySettings: [
      { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
      { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
      { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
      { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
    ],
  });

  const responseText = (result as any)?.candidates?.[0]?.content?.parts?.[0]?.text as string | undefined;

  if (!responseText) {
    const feedback = (result as any)?.promptFeedback;
    if (feedback?.blockReason) throw new Error(`Gemini safety blocked the request: ${feedback.blockReason}`);
    throw new Error('Gemini returned an empty response for prompt generation.');
  }

  const raw = extractPossibleJSON(responseText);
  let parsed: {
    analysis?: { backdrop?: string; tone?: string };
    scenes?: Array<{ scene_id?: number | string; scene?: number | string; id?: number | string; index?: number | string; prompt: string }>;
  };
  try {
    parsed = parseModelJSON(raw);
  } catch (e: any) {
    console.error('Failed to parse JSON from Gemini:', responseText);
    throw new Error(`Gemini returned malformed JSON: ${e.message}`);
  }

  if (!parsed?.analysis || !Array.isArray(parsed.scenes)) {
    console.error('Bad JSON payload from model:', raw);
    throw new Error('Gemini returned malformed JSON for prompt generation.');
  }

  console.log('Gemini Analysis complete:', parsed.analysis);

  const promptMap = new Map<number, string>();

  parsed.scenes.forEach((scenePrompt: any) => {
    const sid = extractSceneId(scenePrompt);
    const pRaw = typeof scenePrompt.prompt === 'string' ? scenePrompt.prompt.trim() : '';
    if (sid && pRaw) {
      const promptFinal = /16:9\.?$/i.test(pRaw)
        ? pRaw
        : (pRaw.endsWith('.') ? `${pRaw} 16:9.` : `${pRaw}. 16:9.`);
      promptMap.set(sid, promptFinal);
    }
  });

  if (promptMap.size === 0) {
    console.warn('No usable scene IDs found in model output; applying positional fallback.');
    parsed.scenes.forEach((scenePrompt: any, idx: number) => {
      const pRaw = typeof scenePrompt.prompt === 'string' ? scenePrompt.prompt.trim() : '';
      if (!pRaw) return;
      const promptFinal = /16:9\.?$/i.test(pRaw)
        ? pRaw
        : (pRaw.endsWith('.') ? `${pRaw} 16:9.` : `${pRaw}. 16:9.`);
      promptMap.set(idx + 1, promptFinal);
    });
  }

  const updatedScenes = scenes.map(scene => ({
    ...scene,
    prompt: promptMap.get(scene.scene) || `Failed to generate prompt for scene ${scene.scene}`,
  }));

  await saveChunkedJson(jobId, updatedScenes);
  console.log('Scene prompts generated and saved.');
  return updatedScenes;
}

// STEP 4: GENERATE IMAGES
async function runStep4_GenerateImages(
  jobId: string,
  scenesWithPrompts: Scene[]
): Promise<Scene[]> {
  console.log(`RUNNING: Step 4 Image Generation for job ${jobId}...`);

  const finalScenes: Scene[] = [];

  for (const scene of scenesWithPrompts) {
    if (!scene.prompt || scene.prompt.startsWith('Failed to generate')) {
      console.log(`Skipping Scene ${scene.scene}: No valid prompt provided.`);
      finalScenes.push(scene);
      continue;
    }

    console.log(`Generating image for Scene ${scene.scene}...`);
    try {
      const request = {
        prompt: scene.prompt,
        config: { numberOfImages: 1, aspectRatio: DEFAULT_AR },
        safetySettings: [
          { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
          { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
          { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
          { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
        ],
      } as const;

      const imgResult = await googleAI.models.generateImages({ model: IMAGE_MODEL, ...request });

      const image = (imgResult as any)?.generatedImages?.[0];
      const imageBytesB64 = image?.image?.imageBytes;

      if (!imageBytesB64) throw new Error('Imagen returned no image data.');
      const imageBytes = Buffer.from(imageBytesB64, 'base64');

      const filename = `scene_${String(scene.scene).padStart(3, '0')}_${Date.now()}.png`;
      const file = storage.bucket(imageBucketName).file(filename);

      await file.save(imageBytes, { contentType: 'image/png' });

      const publicUrl = `https://storage.googleapis.com/${imageBucketName}/${filename}`;
      console.log(`Scene ${scene.scene} image saved to: ${publicUrl}`);

      finalScenes.push({ ...scene, image_url: publicUrl });

      console.log(`Waiting ${SLEEP_BETWEEN_IMAGES_SEC}s before next image...`);
      await sleep(SLEEP_BETWEEN_IMAGES_SEC * 1000);
    } catch (error: any) {
      console.error(`Failed to generate image for Scene ${scene.scene}:`, error?.message || error);
      finalScenes.push({
        ...scene,
        image_url: `ERROR: ${error?.message || 'Unknown image generation error'}`,
      });
      console.log(`Error encountered. Waiting ${SLEEP_BETWEEN_IMAGES_SEC}s...`);
      await sleep(SLEEP_BETWEEN_IMAGES_SEC * 1000);
    }
  }

  await saveChunkedJson(jobId, finalScenes);
  console.log('Image generation complete. Final JSON saved.');
  return finalScenes;
}

// STEP 5: GENERATE VIDEO
async function runStep5_CreateVideo(
  jobId: string,
  scenes: Scene[],
  gcsAudioUri: string
): Promise<string> {
  await assertFfmpeg();

  console.log(`RUNNING: Step 5 Video Creation for job ${jobId}...`);
  const TEMP_DIR = path.join(process.cwd(), 'temp', jobId);
  await fs.mkdir(TEMP_DIR, { recursive: true });

  const finalVideoName = `${jobId}.mp4`;
  const finalVideoPath = path.join(TEMP_DIR, finalVideoName);
  const inputsTxtPath = path.join(TEMP_DIR, 'inputs.txt');
  const localAudioPath = path.join(TEMP_DIR, 'audio.wav'); // adjust if source isn’t wav

  try {
    console.log('Downloading assets...');

    const [audioBucket, ...audioFileParts] = gcsAudioUri.replace('gs://', '').split('/');
    const audioFileName = audioFileParts.join('/');
    await downloadGCSFile(audioBucket, audioFileName, localAudioPath);

    let inputsTxtContent = '';
    const localImagePaths: string[] = [];
    let lastImageFileName: string | null = null;

    for (const scene of scenes) {
      if (!scene.image_url) continue;

      const imageUrl = new URL(scene.image_url);
      const imageFileName = path.basename(imageUrl.pathname);
      const localImagePath = path.join(TEMP_DIR, imageFileName);

      await downloadGCSFile(imageBucketName, imageFileName, localImagePath);

      const duration = Math.max(0.05, scene.end_time_secs - scene.start_time_secs);

      inputsTxtContent += `file '${imageFileName}'\n`;
      inputsTxtContent += `duration ${duration.toFixed(3)}\n`;

      lastImageFileName = imageFileName;
      localImagePaths.push(localImagePath);
    }

    // Concat demuxer requires the last file listed again without duration
    if (lastImageFileName) {
      inputsTxtContent += `file '${lastImageFileName}'\n`;
    }

    await fs.writeFile(inputsTxtPath, inputsTxtContent, 'utf-8');
    console.log('Assets downloaded and inputs.txt created.');

    console.log('Starting FFmpeg...');
    await new Promise((resolve, reject) => {
      ffmpeg()
        .input(inputsTxtPath)
        .inputOptions(['-f', 'concat', '-safe', '0'])
        .input(localAudioPath)
        .outputOptions('-c:v', 'libx264')
        .outputOptions('-c:a', 'aac')
        .outputOptions('-shortest')
        .outputOptions('-pix_fmt', 'yuv420p')
        .outputOptions('-r', '24')
        .on('start', (cmd) => console.log('FFmpeg command:', cmd))
        .on('error', (err) => reject(new Error(`FFmpeg error: ${err.message}`)))
        .on('end', () => resolve(null))
        .save(finalVideoPath);
    });

    console.log(`Video created: ${finalVideoPath}`);

    console.log(`Uploading final video to gs://${videoBucketName}/${finalVideoName}`);
    await storage.bucket(videoBucketName).upload(finalVideoPath, {
      destination: finalVideoName,
      contentType: 'video/mp4',
    });
    const publicUrl = `https://storage.googleapis.com/${videoBucketName}/${finalVideoName}`;

    console.log(`Upload complete! Video available at: ${publicUrl}`);
    return publicUrl;
  } catch (error) {
    console.error('Error in Step 5 (CreateVideo):', error);
    throw error;
  } finally {
    console.log(`Cleaning up temp directory: ${TEMP_DIR}`);
    await fs.rm(TEMP_DIR, { recursive: true, force: true });
  }
}

// --- 6. THE MAIN SERVER ACTION ---
export async function generateStoryboard(formData: FormData) {
  try {
    // ===================================================================
    // =================== DEBUG: START FROM STEP 5 =========================
    // ===================================================================
    // console.warn('!!!!!!!!!! RUNNING IN DEBUG MODE: STARTING FROM STEP 5 !!!!!!!!!!!');

    // const debugJobId = 'final_1764653265533';
    // const debugChunkedJsonFile = 'final_1764653265533.json';

    // console.log(`DEBUG: Fetching chunked JSON: gs://${chunkedBucketName}/${debugChunkedJsonFile}`);

    // const jsonFile = storage.bucket(chunkedBucketName).file(debugChunkedJsonFile);
    // const [jsonData] = await jsonFile.download();

    // const debugFinalScenes = JSON.parse(jsonData.toString('utf-8')) as Scene[];
    // console.log(`DEBUG: Loaded ${debugFinalScenes.length} scenes from JSON.`);

    // if (debugFinalScenes.length === 0) {
    //   throw new Error('Debug JSON file was empty or invalid.');
    // }

    // // We need the audio file. Assuming .wav. If your file is .mp3, change this line.
    // const debugGcsAudioUri = `gs://${audioBucketName}/${debugJobId}.wav`;

    // console.log('DEBUG: Running Step 5 (Create Video)...');
    // const debugVideoUrl = await runStep5_CreateVideo(debugJobId, debugFinalScenes, debugGcsAudioUri);

    // console.log('--- DEBUG STORYBOARD PIPELINE COMPLETE (Step 5) ---');
    // return { scenes: debugFinalScenes, video: debugVideoUrl };

    // ===================================================================
    // ================= END DEBUG BLOCK =================================
    // ===================================================================

    // --- PRODUCTION FLOW (unreachable while debug block returns above) ---
    const file = formData.get('audioFile') as File;
    if (!file) throw new Error('No audio file found.');
    const jobIdBase = getJobId(file.name);
    const fullJobId = `${jobIdBase}_${Date.now()}`;
    console.log(`Starting new job: ${fullJobId}`);
    const { words, gcsAudioUri } = await runStep1_Transcribe(fullJobId, file);
    const fullScript = words.map(w => w.word).join(' ');
    const scenes = await runStep2_ChunkScript(fullJobId, words);
    const scenesWithPrompts = await runStep3_AnalyzeAndGeneratePrompts(fullJobId, scenes, fullScript);
    const finalScenes = await runStep4_GenerateImages(fullJobId, scenesWithPrompts);
    const videoUrl = await runStep5_CreateVideo(fullJobId, finalScenes, gcsAudioUri);
    console.log('--- STORYBOARD PIPELINE COMPLETE ---');
    return { scenes: finalScenes, video: videoUrl };
  } catch (error: any) {
    console.error('Error in generateStoryboard:', error);
    return { error: error?.message || 'An unknown server error occurred.' };
  }
}
