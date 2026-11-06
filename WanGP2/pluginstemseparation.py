"""Stem Separator plugin for WanGP.

Takes an mp3 / wav (other formats ffmpeg can read also work), runs
htdemucs_6s (optionally the guitar fine-tune from
adityalakhani/htdemucs-6s-guitar-ft) and writes one wav per stem plus a zip.
"""
import os
import shutil
import subprocess
import tempfile
import time

import gradio as gr

from shared.gradio.progress import WangpProgress
from shared.utils.plugins import WAN2GPPlugin
from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources

PlugIn_Name = "Stem Separator"
PlugIn_Id = "StemSeparatorPlugin"

STEM_NAMES = ["drums", "bass", "other", "vocals", "guitar", "piano"]
SAMPLE_RATE = 44100
HF_REPO = "adityalakhani/htdemucs-6s-guitar-ft"
HF_FILE = "guitar_htdemucs_6s.pt"
MODEL_GUITAR_FT = "Guitar fine-tune (htdemucs-6s-guitar-ft)"
MODEL_STOCK = "Stock htdemucs_6s (Meta AI)"
AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]


def _output_root():
    # Same place WanGP puts generated audio, in a "stems" subfolder.
    try:
        import wgp
        base = getattr(wgp, "audio_save_path", None) or "outputs"
    except Exception:
        base = "outputs"
    return os.path.join(base, "stems")


def _unique_dir(root, name):
    path = os.path.join(root, name)
    n = 2
    while os.path.exists(path):
        path = os.path.join(root, f"{name}_{n}")
        n += 1
    os.makedirs(path)
    return path


def _extract_audio(src, dst):
    """Decode anything ffmpeg understands to 44.1 kHz stereo wav."""
    if shutil.which("ffmpeg") is None:
        raise gr.Error("ffmpeg was not found on this machine.")
    cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-ar", str(SAMPLE_RATE), "-ac", "2", dst]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise gr.Error("ffmpeg could not read this audio file. See the console for details.")


def _load_model(choice, device):
    import torch
    try:
        from demucs.pretrained import get_model
    except ImportError:
        raise gr.Error("demucs is not installed. Run:  pip install demucs==4.0.1 huggingface_hub soundfile  then retry.")

    model = get_model("htdemucs_6s")  # also downloads Meta's stock weights
    if choice == MODEL_GUITAR_FT:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = {f"models.0.{k}": v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def separate(state, audio_file, model_choice, stems_to_save, shifts, overlap, progress=WangpProgress()):
    if not audio_file:
        raise gr.Error("Please upload an mp3 or wav file first.")
    if not stems_to_save:
        raise gr.Error("Select at least one stem to save.")

    import torch
    import numpy as np
    import soundfile as sf
    from demucs.apply import apply_model

    audio_file = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", str(audio_file))
    track_name = os.path.splitext(os.path.basename(audio_file))[0]
    out_dir = _unique_dir(_output_root(), track_name)

    # acquire_GPU_ressources pauses/releases WanGP's own model so we get the VRAM.
    acquire_GPU_ressources(state, PlugIn_Id, PlugIn_Name, gr=gr)
    model = None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            cap = torch.cuda.get_device_capability(0)
            if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
                print("[StemSeparator] GPU architecture not supported by this PyTorch build, using CPU.")
                device = "cpu"

        progress(0.02, desc="Decoding audio...")
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "input.wav")
            _extract_audio(audio_file, wav_path)
            data, _ = sf.read(wav_path, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(data.T)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        duration = waveform.shape[1] / SAMPLE_RATE

        progress(0.10, desc=f"Loading model ({'GPU' if device == 'cuda' else 'CPU'})...")
        model = _load_model(model_choice, device)

        progress(0.20, desc=f"Separating {duration:.0f}s of audio (can take a few minutes)...")
        start = time.time()
        with torch.no_grad():
            sources = apply_model(
                model,
                waveform.unsqueeze(0).to(device),
                shifts=int(shifts),
                overlap=float(overlap),
                progress=True,
            )
        sources = sources[0].cpu()
        print(f"[StemSeparator] separation finished in {time.time() - start:.1f}s")

        progress(0.90, desc="Saving stems...")
        saved = {}
        for name in STEM_NAMES:
            if name not in stems_to_save:
                continue
            path = os.path.join(out_dir, f"{name}.wav")
            sf.write(path, sources[STEM_NAMES.index(name)].T.numpy(), samplerate=SAMPLE_RATE, subtype="PCM_24")
            saved[name] = path

        zip_path = shutil.make_archive(os.path.join(out_dir, f"{track_name}_stems"), "zip", out_dir)
        progress(1.0, desc="Done")
        gr.Info(f"Stems saved to {out_dir}")
    finally:
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        release_GPU_ressources(state, PlugIn_Id)

    return [saved.get(n) for n in STEM_NAMES] + [zip_path, f"Saved to: {os.path.abspath(out_dir)}"]


class StemSeparatorPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.0.0"
        self.description = "Split an mp3/wav into drums, bass, other, vocals, guitar and piano stems."

    def setup_ui(self):
        self.request_component("state")
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)

    def create_ui(self, api_session=None):
        state = self.state
        with gr.Column():
            gr.Markdown(
                "### Stem Separator\n"
                "Upload an **mp3** or **wav**, pick the stems you want, and click *Separate*. "
                "Uses the `htdemucs_6s` architecture (drums, bass, other, vocals, guitar, piano)."
            )
            audio_in = gr.File(label="Audio file (mp3 / wav)", file_types=AUDIO_EXTENSIONS, type="filepath")
            preview = gr.Audio(label="Preview", interactive=False)
            with gr.Row():
                model_choice = gr.Dropdown(
                    [MODEL_GUITAR_FT, MODEL_STOCK], value=MODEL_GUITAR_FT, label="Model"
                )
                stems = gr.CheckboxGroup(STEM_NAMES, value=STEM_NAMES, label="Stems to save")
            with gr.Row():
                shifts = gr.Slider(1, 5, value=1, step=1, label="Shifts (higher = better quality, slower)")
                overlap = gr.Slider(0.1, 0.5, value=0.25, step=0.05, label="Overlap")
            run_btn = gr.Button("Separate", variant="primary")
            progress_ui = WangpProgress.component()
            status = gr.Text(label="Output folder", interactive=False)
            with gr.Row():
                outs = [gr.Audio(label=n.capitalize(), interactive=False) for n in STEM_NAMES[:3]]
            with gr.Row():
                outs += [gr.Audio(label=n.capitalize(), interactive=False) for n in STEM_NAMES[3:]]
            zip_out = gr.File(label="Download all (zip)")

        audio_in.change(lambda f: f, inputs=[audio_in], outputs=[preview], queue=False)
        WangpProgress.bind(
            run_btn.click,
            separate,
            inputs=[state, audio_in, model_choice, stems, shifts, overlap],
            outputs=outs + [zip_out, status],
            component=progress_ui,
        )
