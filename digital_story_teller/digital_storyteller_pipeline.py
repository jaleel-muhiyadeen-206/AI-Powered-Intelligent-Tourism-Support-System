"""
=============================================================================
  DIGITAL STORYTELLER ENGINE — Full Pipeline
  AI-Powered Tourism Support System for Sri Lanka | Group 17
=============================================================================
  Component:  Process 02 — Digital Storyteller Engine
  Author:     Group 17 (Storyteller Module)
  Models:     Ollama (llama3.1) — Scene Script Generation
              Google Veo 3.1  — Video Generation (8s clips x 7 = ~56s doc)
  Pipeline:   EDA -> Preprocess -> Scene Planning -> Video Gen (with native audio) -> Assembly
=============================================================================

  DEPENDENCIES:
    pip install pandas ollama google-genai moviepy pillow requests
                matplotlib seaborn gTTS

  FOLDER STRUCTURE EXPECTED:
    landmark_references/
      Sigiriya/          <- JPG/PNG reference images for each landmark
      Nine_Arches_Bridge_Ella/
      ...
    sri_lanka_landmarks_final.csv

  OUTPUTS (auto-created):
    outputs/
      eda_report/        <- EDA charts + CSV summary
      cleaned_references/<- Preprocessed 16:9 720p images
      scene_plans/       <- JSON scene scripts per landmark
      raw_clips/         <- Individual Veo-generated MP4 clips
      final_videos/      <- Final assembled documentary MP4s
=============================================================================
"""

import os
import ast
import json
import time
import glob
import logging
import requests
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from moviepy import VideoFileClip, concatenate_videoclips, AudioFileClip
from google import genai
from google.genai import types
import ollama

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
#  CONFIGURATION  -- Edit these before running
# =============================================================================
GEMINI_API_KEY   = "AIzaSyDVQDqxIpP-pldmVXkmiISaMi_HeIVuXkc"   # <-- Replace with your key
VEO_MODEL        = "veo-3.1-fast-generate-preview"
OLLAMA_MODEL     = "llama3.1"
CSV_PATH         = "sri_lanka_landmarks_final.csv"
IMG_INPUT_DIR    = "reference_images"          # Your reference image folders

# Pipeline output directories
EDA_DIR          = "outputs/eda_report"
CLEAN_DIR        = "outputs/cleaned_references"
PLAN_DIR         = "outputs/scene_plans"
CLIPS_DIR        = "outputs/raw_clips"
FINAL_DIR        = "outputs/final_videos"

# Video settings
SCENES_PER_DOC   = 7          # 7 x 8s = 56s documentary
SCENE_DURATION_S = 8          # Veo 3.1 max per clip
CLIP_RESOLUTION  = (1280, 720) # 720p 16:9
VEO_ASPECT_RATIO = "16:9"
POLL_INTERVAL_S  = 15          # Seconds between Veo status checks


# =============================================================================
#  UTILITY HELPERS
# =============================================================================

def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def safe_name(text: str) -> str:
    """Convert landmark name to a filesystem-safe string."""
    return (text.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("'", ""))


# =============================================================================
#  STAGE 0 -- LOAD & VALIDATE DATA
# =============================================================================

def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV, parse the Facts list column, and validate integrity."""
    log.info("Loading dataset from: %s", path)
    df = pd.read_csv(path)

    # Parse stringified Python list in 'Facts' column
    df["Facts"] = df["Facts"].apply(ast.literal_eval)

    # Validation
    required_cols = {"Landmark", "Facts", "Significance"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    null_counts = df.isnull().sum()
    if null_counts.any():
        log.warning("Null values found:\n%s", null_counts[null_counts > 0])

    df["facts_count"]      = df["Facts"].apply(len)
    df["significance_len"] = df["Significance"].apply(len)
    df["landmark_safe"]    = df["Landmark"].apply(safe_name)

    log.info("Dataset loaded: %d landmarks, columns: %s", len(df), df.columns.tolist())
    return df


# =============================================================================
#  STAGE 1 -- EDA (Exploratory Data Analysis)
# =============================================================================

def run_eda(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Generates EDA summary statistics and charts.
    Returns the enriched dataframe with additional EDA columns.
    """
    make_dirs(out_dir)
    log.info("Running EDA -> %s", out_dir)

    # Text length distributions
    df["significance_words"] = df["Significance"].apply(lambda x: len(x.split()))
    df["facts_total_words"]  = df["Facts"].apply(
        lambda lst: sum(len(f.split()) for f in lst)
    )
    df["avg_fact_words"] = df["Facts"].apply(
        lambda lst: round(np.mean([len(f.split()) for f in lst]), 1)
    )

    # Image availability per landmark
    img_counts = []
    for _, row in df.iterrows():
        pattern = os.path.join(IMG_INPUT_DIR, f"*{row['landmark_safe']}*")
        folders = glob.glob(pattern)
        count = 0
        if folders:
            imgs = (glob.glob(os.path.join(folders[0], "*.jpg")) +
                    glob.glob(os.path.join(folders[0], "*.png")) +
                    glob.glob(os.path.join(folders[0], "*.jpeg")))
            count = len(imgs)
        img_counts.append(count)
    df["reference_images"] = img_counts

    # Save EDA summary CSV
    eda_summary_cols = [
        "Landmark", "facts_count", "significance_words",
        "facts_total_words", "avg_fact_words", "reference_images"
    ]
    eda_df = df[eda_summary_cols].copy()
    eda_df.to_csv(os.path.join(out_dir, "eda_summary.csv"), index=False)
    log.info("EDA summary CSV saved.")

    # ---- Chart 1: 2x2 Overview Dashboard ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EDA -- Sri Lanka Landmarks Dataset", fontsize=15, fontweight="bold")

    # Facts word count per landmark (horizontal bar)
    axes[0, 0].barh(df["Landmark"], df["facts_total_words"], color="#2196F3")
    axes[0, 0].set_title("Total Words in Facts per Landmark")
    axes[0, 0].set_xlabel("Word Count")
    axes[0, 0].tick_params(axis="y", labelsize=7)

    # Significance length distribution (histogram)
    axes[0, 1].hist(df["significance_words"], bins=10, color="#4CAF50", edgecolor="white")
    axes[0, 1].set_title("Distribution of Significance Description Length (Words)")
    axes[0, 1].set_xlabel("Word Count")
    axes[0, 1].set_ylabel("Frequency")

    # Reference images per landmark (bar)
    axes[1, 0].bar(range(len(df)), df["reference_images"], color="#FF9800", edgecolor="white")
    axes[1, 0].set_xticks(range(len(df)))
    axes[1, 0].set_xticklabels(df["Landmark"], rotation=90, fontsize=6)
    axes[1, 0].set_title("Reference Images Available per Landmark")
    axes[1, 0].set_ylabel("Image Count")

    # Average fact word count (boxplot)
    axes[1, 1].boxplot(df["avg_fact_words"], patch_artist=True,
                       boxprops=dict(facecolor="#E91E63", color="#333"))
    axes[1, 1].set_title("Average Words per Fact (across all landmarks)")
    axes[1, 1].set_ylabel("Avg Words per Fact")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_overview.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Chart 2: Content Coverage Heatmap ----
    fig2, ax2 = plt.subplots(figsize=(8, 9))
    heat_data = df[["facts_count", "significance_words",
                    "facts_total_words", "reference_images"]].copy()
    heat_data.index = df["Landmark"]
    sns.heatmap(heat_data, annot=True, fmt="g", cmap="YlOrRd", ax=ax2, linewidths=0.3)
    ax2.set_title("Content Coverage Heatmap per Landmark", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coverage_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("EDA charts saved.")

    # Print EDA summary
    print("\n" + "=" * 60)
    print("  EDA REPORT -- Sri Lanka Landmarks Dataset")
    print("=" * 60)
    print(f"  Total Landmarks   : {len(df)}")
    print(f"  Facts per Landmark: {df['facts_count'].unique()[0]} (uniform)")
    print(f"  Avg Words/Fact    : {df['avg_fact_words'].mean():.1f}")
    print(f"  Significance Avg  : {df['significance_words'].mean():.1f} words")
    print(f"  Total Ref Images  : {df['reference_images'].sum()}")
    print(f"  Missing Images    : {(df['reference_images'] == 0).sum()} landmarks")
    print("=" * 60 + "\n")

    return df


# =============================================================================
#  STAGE 2 -- IMAGE PREPROCESSING
# =============================================================================

def preprocess_images(df: pd.DataFrame, input_dir: str, output_dir: str) -> dict:
    """
    For each landmark, finds its reference image folder and:
      1. Converts all images to RGB
      2. Center-crops to 16:9 aspect ratio
      3. Resizes to 1280x720 (720p)
      4. Saves as high-quality JPEG
    Returns a dict mapping landmark name -> [list of cleaned image paths].
    """
    make_dirs(output_dir)
    log.info("Preprocessing images -> %s", output_dir)
    image_map = {}

    for _, row in df.iterrows():
        lm     = row["Landmark"]
        safe   = row["landmark_safe"]
        target = os.path.join(output_dir, safe)
        make_dirs(target)

        # Search for matching folder (partial match on safe name)
        pattern = os.path.join(input_dir, f"*{safe}*")
        folders = glob.glob(pattern)

        if not folders:
            log.warning("No image folder found for: %s (skipping)", lm)
            image_map[lm] = []
            continue

        src_folder = folders[0]
        raw_imgs   = (glob.glob(os.path.join(src_folder, "*.jpg"))  +
                      glob.glob(os.path.join(src_folder, "*.jpeg")) +
                      glob.glob(os.path.join(src_folder, "*.png")))

        if not raw_imgs:
            log.warning("No images in folder: %s", src_folder)
            image_map[lm] = []
            continue

        cleaned = []
        for img_path in sorted(raw_imgs):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    w, h = img.size

                    # Center crop to 16:9
                    target_ratio  = 16 / 9
                    current_ratio = w / h
                    if current_ratio > target_ratio:        # Too wide
                        new_w = int(h * target_ratio)
                        left  = (w - new_w) // 2
                        img   = img.crop((left, 0, left + new_w, h))
                    else:                                   # Too tall
                        new_h = int(w / target_ratio)
                        top   = (h - new_h) // 2
                        img   = img.crop((0, top, w, top + new_h))

                    # Resize to 720p
                    img = img.resize(CLIP_RESOLUTION, Image.LANCZOS)

                    out_name = os.path.splitext(os.path.basename(img_path))[0] + ".jpg"
                    out_path = os.path.join(target, out_name)
                    img.save(out_path, "JPEG", quality=90)
                    cleaned.append(out_path)

            except Exception as e:
                log.error("Failed to process %s: %s", img_path, e)

        image_map[lm] = cleaned
        log.info("  %-42s -> %d images processed", lm, len(cleaned))

    return image_map


def preprocess_eda(image_map: dict, out_dir: str):
    """
    Post-preprocessing EDA: validates cleaned images (dimensions, aspect
    ratio, file size) and saves a quality-check chart.
    """
    stats = []
    for landmark, paths in image_map.items():
        for p in paths:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    stats.append({
                        "landmark": landmark,
                        "width": w, "height": h,
                        "aspect_ratio": round(w / h, 3),
                        "size_kb": os.path.getsize(p) / 1024
                    })
            except Exception:
                pass

    if not stats:
        log.warning("No cleaned images found for post-preprocess EDA.")
        return

    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(os.path.join(out_dir, "cleaned_image_stats.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Post-Preprocessing Image Quality Validation", fontsize=12, fontweight="bold")

    axes[0].hist(df_stats["aspect_ratio"], bins=15, color="#00BCD4", edgecolor="white")
    axes[0].set_title("Aspect Ratio Distribution")
    axes[0].set_xlabel("Ratio (target = 1.778)")
    axes[0].axvline(16/9, color="red", linestyle="--", label="16:9 target")
    axes[0].legend()

    axes[1].hist(df_stats["size_kb"], bins=15, color="#8BC34A", edgecolor="white")
    axes[1].set_title("File Size Distribution (KB)")
    axes[1].set_xlabel("Size (KB)")

    axes[2].scatter(df_stats["width"], df_stats["height"], alpha=0.6, color="#FF5722")
    axes[2].set_title("Image Dimensions Scatter")
    axes[2].set_xlabel("Width (px)")
    axes[2].set_ylabel("Height (px)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cleaned_image_eda.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Post-preprocessing image EDA chart saved.")


# =============================================================================
#  STAGE 3 -- SCENE PLANNING WITH LLM (Ollama / llama3.1)
# =============================================================================

SCENE_PLAN_PROMPT = """[SYSTEM: ACT AS AN AWARD-WINNING CINEMATIC DOCUMENTARY DIRECTOR.]

You are crafting a 56-second narrated video documentary about a Sri Lankan landmark.
The documentary consists of exactly 7 scenes, each lasting 8 seconds.

Narrative arc for the 7 scenes:
  Scene 1: Dramatic establishing aerial/wide shot + hook narration
  Scene 2: Historical origin story
  Scene 3: Key architectural or natural feature highlight
  Scene 4: Cultural or spiritual significance
  Scene 5: Unique fact or "wow" moment
  Scene 6: Modern-day visitor experience
  Scene 7: Cinematic closing shot + inspiring call-to-action

LANDMARK: {landmark}
SIGNIFICANCE: {significance}
KEY FACTS:
{facts_block}

OUTPUT FORMAT (strict -- exactly 7 lines, pipe-delimited):
Scene 1 | Video: [cinematic visual description] | Audio: [spoken narration, 20-25 words]
Scene 2 | Video: [cinematic visual description] | Audio: [spoken narration, 20-25 words]
Scene 3 | Video: [cinematic visual description] | Audio: [spoken narration, 20-25 words]
Scene 4 | Video: [cinematic visual description] | Audio: [spoken narration, 20-25 words]
Scene 5 | Video: [cinematic visual description] | Audio: [spoken narration, 20-25 words]
Scene 6 | Video: [cinematic visual description] | Audio: [spoken narration, 20-25 words]
Scene 7 | Video: [cinematic visual description] | Audio: [spoken narration, 20-25 words]

RULES:
- Each Video prompt must be a rich paragraph describing colours, camera angle, lighting
- Each Audio must be complete evocative narration (no ellipsis, no placeholders)
- Output ONLY the 7 scene lines, nothing else
"""


def parse_scene_plan(raw_text: str) -> list:
    """
    Parses the LLM output into a list of scene dicts.
    Each dict: {scene_num, video_prompt, audio_narration}
    """
    scenes = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line.startswith("Scene"):
            continue
        try:
            parts = [p.strip() for p in line.split("|")]
            scene_num = int(parts[0].replace("Scene", "").strip())
            video = parts[1].replace("Video:", "").strip()
            audio = parts[2].replace("Audio:", "").strip()
            scenes.append({
                "scene_num": scene_num,
                "video_prompt": video,
                "audio_narration": audio
            })
        except Exception:
            log.warning("Could not parse scene line: %s", line)

    scenes = scenes[:SCENES_PER_DOC]
    if len(scenes) < SCENES_PER_DOC:
        log.warning("Only %d scenes parsed (expected %d)", len(scenes), SCENES_PER_DOC)
    return scenes


def generate_scene_plan(row: pd.Series, plan_dir: str) -> list:
    """
    Calls Ollama llama3.1 to generate 7 scene plans for a landmark.
    Caches the result to JSON to avoid re-running on pipeline restarts.
    Returns the parsed list of scene dicts.
    """
    make_dirs(plan_dir)
    safe  = row["landmark_safe"]
    lm    = row["Landmark"]
    cache = os.path.join(plan_dir, f"{safe}_scene_plan.json")

    if os.path.exists(cache):
        log.info("  Using cached scene plan: %s", cache)
        with open(cache) as f:
            return json.load(f)

    facts_block = "\n".join(f"  - {fact}" for fact in row["Facts"])
    prompt = SCENE_PLAN_PROMPT.format(
        landmark=lm,
        significance=row["Significance"],
        facts_block=facts_block
    )

    log.info("  Calling llama3.1 for scene plan: %s ...", lm)
    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    raw_text = response["response"]

    scenes = parse_scene_plan(raw_text)

    with open(cache, "w") as f:
        json.dump(scenes, f, indent=2)
    log.info("  Scene plan cached -> %s (%d scenes)", cache, len(scenes))

    return scenes


# =============================================================================
#  STAGE 4 -- VEO 3.1 VIDEO GENERATION
# =============================================================================

def download_veo_video(uri: str, dest_path: str, api_key: str) -> bool:
    """Download a Veo-generated video from its URI."""
    headers = {"X-Goog-Api-Key": api_key}
    try:
        response = requests.get(uri, headers=headers, stream=True, timeout=60)
        if response.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        log.error("Download failed (HTTP %d): %s", response.status_code, uri)
    except Exception as e:
        log.error("Download exception: %s", e)
    return False


def _load_image_as_veo_image(image_path: str):
    """
    Reads an image file from disk and returns a types.Image object with
    bytesBase64Encoded + mimeType, which is what the Veo API requires.
    Returns None if loading fails.
    """
    import base64
    import mimetypes

    mime, _ = mimetypes.guess_type(image_path)
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        mime = "image/jpeg"   # safe fallback for .jpg files

    try:
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
        b64_str = base64.b64encode(raw_bytes).decode("utf-8")
        return types.Image(
            image_bytes=base64.b64decode(b64_str),   # pass raw bytes
            mime_type=mime,
        )
    except Exception as e:
        log.error("    Failed to load reference image %s: %s", image_path, e)
        return None


def generate_veo_clip(
    client,
    scene: dict,
    ref_image_path,
    dest_path: str,
    api_key: str
) -> bool:
    """
    Calls Veo 3.1 to generate a single 8-second video clip for one scene.
    The narration text is embedded directly in the prompt so Veo generates
    native voiceover audio — perfectly synchronised with visuals, no gTTS needed.

    FIX: Veo API requires image as types.Image(image_bytes=..., mime_type=...)
         NOT as a client.files.upload() File object (causes 400 INVALID_ARGUMENT).
    Returns True on success.
    """
    first_frame = None
    if ref_image_path and os.path.exists(ref_image_path):
        log.info("    Loading reference image: %s", os.path.basename(ref_image_path))
        first_frame = _load_image_as_veo_image(ref_image_path)
        if first_frame:
            log.info("    Reference image loaded OK (mime: %s)", first_frame.mime_type)

    # Narration text is embedded in the prompt so Veo generates native audio
    # that is perfectly synced to the visuals — no separate gTTS merge needed.
    veo_prompt = (
        "Cinematic documentary style, 4K quality, smooth camera movement. "
        f"{scene['video_prompt']} "
        "Natural lighting, golden hour tones, no text overlays, no jump cuts, "
        "professional colour grading. "
        f"Voiceover narration spoken by a calm documentary narrator: "
        f"\"{scene['audio_narration']}\""
    )

    log.info("    Submitting Veo job for Scene %d ...", scene["scene_num"])

    operation = client.models.generate_videos(
        model=VEO_MODEL,
        prompt=veo_prompt,
        image=first_frame,           # types.Image or None — both accepted by Veo
        config=types.GenerateVideosConfig(
            aspect_ratio=VEO_ASPECT_RATIO,
            duration_seconds=SCENE_DURATION_S,
        )
    )

    # Poll until Veo finishes
    retries = 0
    max_retries = 40   # 40 x 15s = 10 minutes max per clip
    while not operation.done:
        if retries >= max_retries:
            log.error("    Veo timed out on Scene %d", scene["scene_num"])
            return False
        log.info("    Rendering Scene %d ... (%ds elapsed)",
                 scene["scene_num"], retries * POLL_INTERVAL_S)
        time.sleep(POLL_INTERVAL_S)
        operation = client.operations.get(operation)
        retries += 1

    try:
        video_uri = operation.result.generated_videos[0].video.uri
        success   = download_veo_video(video_uri, dest_path, api_key)
        if success:
            log.info("    Scene %d saved -> %s", scene["scene_num"], dest_path)
        return success
    except Exception as e:
        log.error("    Veo result extraction failed: %s", e)
        return False


# =============================================================================
#  STAGE 5 -- TTS AUDIO GENERATION (gTTS)
# =============================================================================

def generate_audio_narrations(scenes: list, audio_dir: str) -> list:
    """
    Generates TTS audio for each scene narration using gTTS.
    Each audio clip matches the narration text for that scene.
    Returns list of audio file paths (None if generation failed).
    Install: pip install gTTS
    """
    make_dirs(audio_dir)
    audio_paths = []

    try:
        from gtts import gTTS
    except ImportError:
        log.warning("gTTS not installed. Run: pip install gTTS")
        return [None] * len(scenes)

    for scene in scenes:
        out_path = os.path.join(audio_dir, f"scene_{scene['scene_num']:02d}_audio.mp3")
        if os.path.exists(out_path):
            log.info("    Audio cached: %s", out_path)
            audio_paths.append(out_path)
            continue
        try:
            tts = gTTS(text=scene["audio_narration"], lang="en", slow=False)
            tts.save(out_path)
            log.info("    TTS audio saved -> %s", out_path)
            audio_paths.append(out_path)
        except Exception as e:
            log.error("    TTS failed for scene %d: %s", scene["scene_num"], e)
            audio_paths.append(None)

    return audio_paths


# =============================================================================
#  STAGE 6 -- MERGE VIDEO + AUDIO PER CLIP
# =============================================================================

def merge_audio_into_clip(video_path: str, audio_path, out_path: str) -> str:
    """
    Overlays TTS audio onto a Veo video clip and writes the merged file.
    Audio is trimmed to match the video duration (8s).
    If no audio path is provided, returns the original video path unchanged.
    """
    if not audio_path or not os.path.exists(audio_path):
        return video_path

    try:
        vid = VideoFileClip(video_path)
        aud = AudioFileClip(audio_path)

        # Trim audio to video duration
        if aud.duration > vid.duration:
            aud = aud.subclipped(0, vid.duration)

        final = vid.with_audio(aud)
        final.write_videofile(out_path, codec="libx264", audio_codec="aac",
                              logger=None, fps=24)
        vid.close()
        aud.close()
        final.close()
        log.info("    Audio merged -> %s", out_path)
        return out_path
    except Exception as e:
        log.error("    Audio merge failed: %s", e)
        return video_path


# =============================================================================
#  STAGE 7 -- FINAL DOCUMENTARY ASSEMBLY
# =============================================================================

def assemble_documentary(clip_paths: list, landmark_name: str, out_dir: str):
    """
    Concatenates all 7 scene clips into a final ~56-second documentary MP4.
    Returns the output path or None on failure.
    """
    make_dirs(out_dir)
    valid_clips = [p for p in clip_paths if p and os.path.exists(p)]

    if not valid_clips:
        log.error("No valid clips to assemble for: %s", landmark_name)
        return None

    log.info("Assembling documentary: %s (%d clips)", landmark_name, len(valid_clips))

    try:
        clips    = [VideoFileClip(c) for c in valid_clips]
        final    = concatenate_videoclips(clips, method="compose")
        out_path = os.path.join(out_dir, f"{safe_name(landmark_name)}_Documentary.mp4")
        final.write_videofile(out_path, codec="libx264", audio_codec="aac",
                              fps=24, logger=None)
        for c in clips:
            c.close()
        final.close()
        log.info("Documentary complete -> %s", out_path)
        return out_path

    except Exception as e:
        log.error("Assembly failed for %s: %s", landmark_name, e)
        return None


# =============================================================================
#  MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_pipeline(landmark_name: str):
    """
    Executes the complete Digital Storyteller pipeline for ONE landmark.

    Stages:
      0. Load & validate CSV
      1. EDA on full dataset
      2. Preprocess reference images (16:9 crop, 720p resize)
      3. Generate 7-scene script via llama3.1 (Ollama)
      4. Generate 7 x 8s video clips via Veo 3.1
         → Narration text embedded in prompt → Veo generates native audio
         → No gTTS, no audio merge step needed
      5. Concatenate Veo clips into final documentary (~56s)

    Args:
        landmark_name: Name matching a row in the CSV (e.g. "Sigiriya")
    """
    make_dirs(EDA_DIR, CLEAN_DIR, PLAN_DIR, CLIPS_DIR, FINAL_DIR)

    # Stage 0: Load
    df = load_dataset(CSV_PATH)

    # Stage 1: EDA
    df = run_eda(df, EDA_DIR)

    # Stage 2: Preprocess
    image_map = preprocess_images(df, IMG_INPUT_DIR, CLEAN_DIR)
    preprocess_eda(image_map, EDA_DIR)

    # Locate target row
    mask = df["Landmark"].str.lower() == landmark_name.lower()
    if not mask.any():
        available = "\n".join(f"  - {n}" for n in df["Landmark"].tolist())
        raise ValueError(f"Landmark '{landmark_name}' not found.\nAvailable:\n{available}")
    row  = df[mask].iloc[0]
    safe = row["landmark_safe"]

    log.info("\n" + "=" * 60)
    log.info("  PIPELINE: %s", row["Landmark"])
    log.info("=" * 60)

    # Stage 3: Scene planning
    scenes = generate_scene_plan(row, PLAN_DIR)
    if not scenes:
        raise RuntimeError(f"Scene plan failed for: {landmark_name}")

    print(f"\n--- Scene Plan: {row['Landmark']} ---")
    for s in scenes:
        print(f"  Scene {s['scene_num']}")
        print(f"    Video : {s['video_prompt'][:90]}...")
        print(f"    Audio : {s['audio_narration']}")

    # Stage 4: Veo 3.1 generation (with native audio embedded in prompt)
    client      = genai.Client(api_key=GEMINI_API_KEY)
    ref_images  = image_map.get(row["Landmark"], [])
    lm_clip_dir = os.path.join(CLIPS_DIR, safe)
    make_dirs(lm_clip_dir)
    veo_clips   = []   # clips already have native Veo audio — no gTTS merge needed

    for i, scene in enumerate(scenes):
        # Naming: _veo.mp4 to make clear these already contain Veo native audio
        clip_path = os.path.join(lm_clip_dir, f"scene_{scene['scene_num']:02d}_veo.mp4")

        if os.path.exists(clip_path):
            log.info("  Scene %d already exists, skipping Veo call.", scene["scene_num"])
            veo_clips.append(clip_path)
            continue

        ref_img = ref_images[i] if i < len(ref_images) else None
        success = generate_veo_clip(client, scene, ref_img, clip_path, GEMINI_API_KEY)
        veo_clips.append(clip_path if success else None)

    # Veo clips already carry native, perfectly-synced narration audio.

    # Stage 7: Assemble documentary directly from Veo clips
    final_path = assemble_documentary(
        [p for p in veo_clips if p],
        row["Landmark"],
        FINAL_DIR
    )

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Landmark        : {row['Landmark']}")
    print(f"  Scenes planned  : {len(scenes)}")
    print(f"  Clips generated : {sum(1 for c in veo_clips if c)}/{len(scenes)}")
    print(f"  Audio source    : Veo 3.1 native (no gTTS)")
    print(f"  Final video     : {final_path or 'FAILED'}")
    estimated_duration = sum(1 for p in veo_clips if p) * SCENE_DURATION_S
    print(f"  Duration        : ~{estimated_duration}s")
    print("=" * 60 + "\n")

    return final_path


def run_pipeline_all():
    """
    Runs the complete pipeline for ALL 23 landmarks.
    EDA and preprocessing execute once; video generation loops per landmark.
    WARNING: Makes up to 23 x 7 = 161 Veo API calls.
    """
    df        = load_dataset(CSV_PATH)
    df        = run_eda(df, EDA_DIR)
    image_map = preprocess_images(df, IMG_INPUT_DIR, CLEAN_DIR)
    preprocess_eda(image_map, EDA_DIR)

    client  = genai.Client(api_key=GEMINI_API_KEY)
    results = {}

    for _, row in df.iterrows():
        lm   = row["Landmark"]
        safe = row["landmark_safe"]
        log.info("\n>>> Processing: %s", lm)
        try:
            scenes      = generate_scene_plan(row, PLAN_DIR)
            ref_images  = image_map.get(lm, [])
            lm_clip_dir = os.path.join(CLIPS_DIR, safe)
            make_dirs(lm_clip_dir)
            veo_clips   = []

            for i, scene in enumerate(scenes):
                clip_path = os.path.join(lm_clip_dir, f"scene_{scene['scene_num']:02d}_veo.mp4")
                if os.path.exists(clip_path):
                    veo_clips.append(clip_path)
                    continue
                ref_img = ref_images[i] if i < len(ref_images) else None
                ok = generate_veo_clip(client, scene, ref_img, clip_path, GEMINI_API_KEY)
                veo_clips.append(clip_path if ok else None)

            # No gTTS or merge — Veo clips carry native audio
            final = assemble_documentary(
                [p for p in veo_clips if p], lm, FINAL_DIR
            )
            results[lm] = final or "FAILED"

        except Exception as e:
            log.error("Pipeline failed for %s: %s", lm, e)
            results[lm] = "ERROR"

    print("\n\n===== BATCH PIPELINE RESULTS =====")
    for lm, path in results.items():
        status = "OK" if path not in ("FAILED", "ERROR") else "FAIL"
        print(f"  [{status}] {lm:<45} -> {path}")
    print("==================================\n")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Single landmark mode (recommended for testing)
    # Change to any landmark name from the CSV
    TARGET_LANDMARK = "Sigiriya"
    run_pipeline(TARGET_LANDMARK)

