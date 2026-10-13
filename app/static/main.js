const form = document.getElementById("worry-form");
const textInput = document.getElementById("text-input");
const audioInput = document.getElementById("audio-input");
const loadingSection = document.getElementById("loading-section");
const resultSection = document.getElementById("result-section");
const crisisSection = document.getElementById("crisis-section");

// --- Live microphone recording (MediaRecorder), so the user can speak
// straight into the page instead of having to record a file elsewhere and
// upload it. The file input above stays as a fallback for browsers/setups
// where mic access isn't available. ---
const micBtn = document.getElementById("mic-btn");
const micStatus = document.getElementById("mic-status");
const micPreview = document.getElementById("mic-preview");
const micClear = document.getElementById("mic-clear");

let mediaRecorder = null;
let recordedChunks = [];
let recordStartedAt = null;
let timerInterval = null;

function formatElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const m = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const s = String(totalSeconds % 60).padStart(2, "0");
  return `${m}:${s}`;
}

// Sends a recorded/uploaded audio blob to the local Whisper model and types
// the recognized text into the textarea -- this is what makes it behave
// like Google's voice typing (text appears, editable) instead of just
// attaching raw audio that gets silently transcribed on final submit.
async function transcribeAndFill(blob, filename) {
  micStatus.textContent = "Transcribing…";
  const formData = new FormData();
  formData.append("audio", blob, filename);
  try {
    const res = await fetch("/transcribe", { method: "POST", body: formData });
    const data = await res.json();
    if (data.error) {
      micStatus.textContent = data.error;
      return;
    }
    const existing = textInput.value.trim();
    textInput.value = existing ? `${existing} ${data.text}` : data.text;
    micStatus.textContent = "Added to the text box — edit it if needed.";
  } catch (err) {
    micStatus.textContent = "Transcription failed: " + err.message;
  }
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    micStatus.textContent = "Microphone recording isn't supported in this browser — use the file upload below instead.";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: "audio/webm" });
      micPreview.src = URL.createObjectURL(blob);
      micPreview.classList.remove("hidden");
      micClear.classList.remove("hidden");
      stream.getTracks().forEach((t) => t.stop());
      transcribeAndFill(blob, "recording.webm");
    };
    mediaRecorder.start();
    recordStartedAt = Date.now();
    micBtn.textContent = "⏹ Stop";
    micPreview.classList.add("hidden");
    micClear.classList.add("hidden");
    timerInterval = setInterval(() => {
      micStatus.textContent = `Recording… ${formatElapsed(Date.now() - recordStartedAt)}`;
    }, 250);
  } catch (err) {
    micStatus.textContent = "Couldn't access the microphone: " + err.message;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  clearInterval(timerInterval);
  micBtn.textContent = "🎤 Record";
}

micBtn.addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
  } else {
    startRecording();
  }
});

micClear.addEventListener("click", () => {
  micPreview.src = "";
  micPreview.classList.add("hidden");
  micClear.classList.add("hidden");
  micStatus.textContent = "";
});

audioInput.addEventListener("change", () => {
  if (audioInput.files[0]) {
    transcribeAndFill(audioInput.files[0], audioInput.files[0].name);
    audioInput.value = "";
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resultSection.classList.add("hidden");
  crisisSection.classList.add("hidden");
  loadingSection.classList.remove("hidden");

  const formData = new FormData();
  formData.append("text", textInput.value);

  try {
    const res = await fetch("/process", { method: "POST", body: formData });
    const data = await res.json();
    loadingSection.classList.add("hidden");

    if (data.error) {
      alert(data.error);
      return;
    }

    if (data.crisis) {
      crisisSection.classList.remove("hidden");
      crisisSection.querySelector(".crisis-box").textContent = data.message;
      return;
    }

    document.getElementById("summary").textContent = data.summary;
    document.getElementById("narrative").textContent = data.narrative;
    document.getElementById("citation").textContent = `Technique: ${data.technique.name} — ${data.technique.citation}`;
    resultSection.classList.remove("hidden");

    const audioPlayer = document.getElementById("audio-player");
    if (data.audio_url) {
      audioPlayer.src = data.audio_url;
      audioPlayer.classList.remove("hidden");
    }

    pollImage(data.image_status_url);
  } catch (err) {
    loadingSection.classList.add("hidden");
    alert("Something went wrong: " + err.message);
  }
});

function pollImage(statusUrl) {
  const img = document.getElementById("result-image");
  const loadingText = document.getElementById("image-loading");
  const interval = setInterval(async () => {
    const res = await fetch(statusUrl);
    const data = await res.json();
    if (data.status === "done") {
      clearInterval(interval);
      img.src = data.image_url;
      img.classList.remove("hidden");
      loadingText.classList.add("hidden");
    } else if (data.status === "error") {
      clearInterval(interval);
      loadingText.textContent = "Image generation failed, but your text and audio are ready above.";
    }
  }, 2000);
}
