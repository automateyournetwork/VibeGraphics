#!/usr/bin/env python3
"""
MCP Server: VibeGraphics

Project VibeGraphics:
  "VibeGraphics" = AI-generated, theme-driven infographics ("VibeGraphics")
  sourced from your GitHub projects and brought to life with
  nano banana (image) + Veo 3 (animation).

Tools:

  GitHub / Planning:
  - vibe_fetch_github(
        repo_url: str,
        branch?: str="main",
        include_readme?: bool=True,
        include_code?: bool=True,
        max_code_files?: int=10,
        max_code_chars_per_file?: int=20000
    )
      -> Fetch README + code snippets from a GitHub repo and save as JSON bundle.

  - vibe_plan_infographic(
        bundle_path: str,
        theme?: str="cartographer",
        tone?: str="optimistic, exploratory, technical-but-accessible",
        model?: str="gemini-2.0-flash"
    )
      -> Use Gemini to design a VibeGraphic spec (layout, copy, prompts).

  Image (nano banana):
  - banana_generate(
        prompt: str,
        input_paths?: list[str]=None,
        out_dir?: str=".",
        model?: str="gemini-2.5-flash-image-preview",
        n?: int=1
    )
      -> Generate image(s) using Gemini image model, optionally guided by input image(s).

  Video (Veo 3):
  - veo_generate_video(
        prompt: str,
        negative_prompt?: str="",
        out_dir?: str=".",
        model?: str="veo-3.1-generate-preview",
        image_path?: str=None,
        aspect_ratio?: str=None,
        resolution?: str=None,
        seed?: int=None,
        poll_seconds?: int=8,
        max_wait_seconds?: int=900
    )
      -> Generate animation video, optionally conditioned on an input image.

  VibeGraphics pipeline helpers:
  - vibe_render_image_from_spec(
        spec_path: str,
        out_dir?: str=".",
        model?: str="gemini-2.5-flash-image-preview",
        n?: int=1
    )
      -> Read VibeGraphic spec and call banana_generate() using spec['imagePrompt'].

  - vibe_animate_from_spec(
        spec_path: str,
        image_path: str,
        out_dir?: str=".",
        model?: str="veo-3.1-generate-preview",
        aspect_ratio?: str=None,
        resolution?: str=None,
        seed?: int=None
    )
      -> Read VibeGraphic spec and call veo_generate_video() using spec['animationPrompt'].

Notes:
- No base64 in responses (optimized for @file attachment flow).
- Pure MCP over stdio (FastMCP).
- External calls:
    - GitHub REST API (public) via `requests`.
    - Google Gemini / Veo via `google-genai` (requires GEMINI_API_KEY).
"""

import os
import sys
import time
import json
import uuid
import logging
from pathlib import Path
from textwrap import shorten
from typing import Dict, Any, Optional, List, Tuple

# Optional deps: requests (for GitHub)
try:
    import requests
except Exception:
    requests = None  # type: ignore

# Optional deps: google-genai (Gemini/Veo)
try:
    from google import genai
    from google.genai import types as gtypes
    import mimetypes
except Exception as e:  # pragma: no cover - optional
    genai = None  # type: ignore
    gtypes = None  # type: ignore
    mimetypes = None  # type: ignore

# ----- Logging to stderr only -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("VibeGraphicsMCP")

# ---------- FastMCP ----------
try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from fastmcp import FastMCP  # type: ignore

# Base directory for VibeGraphics bundles/specs
VIBE_BASE_DIR = Path(os.path.expanduser("~/.vibegraphics"))
VIBE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GitHub helpers (for vibe_fetch_github)
# =============================================================================

def _normalize_github_repo(repo_url: str) -> Optional[Tuple[str, str]]:
    """
    Given a GitHub URL, return (owner, repo) or None.

    Handles:
      - https://github.com/owner/repo
      - https://github.com/owner/repo/
      - git@github.com:owner/repo.git
    """
    repo_url = repo_url.strip()

    # HTTPS form
    if "github.com" in repo_url and "://" in repo_url:
        # Something like /owner/repo or /owner/repo/...
        parts = repo_url.split("github.com", 1)[1].strip("/")
        segs = parts.split("/")
        if len(segs) >= 2:
            owner, repo = segs[0], segs[1]
            repo = repo.replace(".git", "")
            return owner, repo

    # SSH form
    if repo_url.startswith("git@github.com:"):
        s = repo_url.replace("git@github.com:", "").strip("/")
        segs = s.split("/")
        if len(segs) >= 2:
            owner, repo = segs[0], segs[1]
            repo = repo.replace(".git", "")
            return owner, repo

    return None


def _github_api_list_files(owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
    """
    Hit GitHub's public contents API for a given path.
    Returns a list of dicts (files/dirs) or [] on error.
    """
    if requests is None:
        log.error("requests library not available")
        return []

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            log.warning("GitHub API %s returned %s", url, resp.status_code)
            return []
        data = resp.json()
        if isinstance(data, dict):
            return [data]
        return data
    except Exception as e:
        log.error("GitHub API error for %s: %s", url, e)
        return []


def _github_fetch_text_file(raw_url: str, max_chars: int = 40000) -> str:
    """
    Fetch a text file from a raw GitHub URL, truncated to max_chars.
    """
    if requests is None:
        return ""
    try:
        resp = requests.get(raw_url, timeout=15)
        if resp.status_code != 200:
            log.warning("Failed to fetch %s (%s)", raw_url, resp.status_code)
            return ""
        txt = resp.text
        if len(txt) > max_chars:
            return txt[:max_chars]
        return txt
    except Exception as e:
        log.error("GitHub fetch error for %s: %s", raw_url, e)
        return ""


def _github_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    """
    Construct a raw.githubusercontent.com URL for a given file.
    """
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path.lstrip('/')}"


# =============================================================================
# MCP server
# =============================================================================

mcp = FastMCP("VibeGraphics MCP")

# =============================================================================
# Banana (text→image / image→image) via Gemini
# =============================================================================

@mcp.tool()
def banana_generate(
    prompt: str,
    input_paths: Optional[List[str]] = None,
    out_dir: str = ".",
    model: str = "gemini-2.5-flash-image-preview",
    n: int = 1,
) -> Dict[str, Any]:
    """
    Generate image(s) from a text prompt, optionally guided by input image(s).
    Saves files to out_dir and returns their paths. No base64 returned.

    Args:
      prompt: Text instruction for the model.
      input_paths: Optional list of image file paths (image-to-image).
      out_dir: Directory to write generated files.
      model: Gemini multimodal image generation model.
      n: Desired number of images (best-effort; stream may emit 1+).
    """
    if genai is None or gtypes is None or mimetypes is None:
        return {"ok": False, "error": "google-genai not installed"}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"ok": False, "error": "GEMINI_API_KEY not set in environment"}

    client = genai.Client(api_key=api_key)

    # Build parts: text + optional input images
    parts: List[gtypes.Part] = [gtypes.Part.from_text(text=prompt)]
    input_paths = input_paths or []
    for p in input_paths:
        try:
            with open(p, "rb") as f:
                data = f.read()
            mt, _ = mimetypes.guess_type(p)
            if not mt:
                mt = "image/jpeg"
            parts.append(gtypes.Part.from_bytes(data=data, mime_type=mt))
        except Exception as e:
            return {"ok": False, "error": f"Failed to read input image '{p}': {e}"}

    contents = [gtypes.Content(role="user", parts=parts)]
    config = gtypes.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

    out_dir_p = Path(os.path.expanduser(out_dir))
    out_dir_p.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    texts: List[str] = []
    file_index = 0

    try:
        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            cand = getattr(chunk, "candidates", None)
            if not cand or not cand[0].content or not cand[0].content.parts:
                # Still may have text tokens
                if getattr(chunk, "text", None):
                    texts.append(chunk.text)
                continue

            part = cand[0].content.parts[0]

            # Any TEXT tokens emitted
            if getattr(chunk, "text", None):
                texts.append(chunk.text)

            # Any IMAGE blobs emitted
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                mt = getattr(inline, "mime_type", "image/png")
                ext = mimetypes.guess_extension(mt) or ".png"
                ts = time.strftime("%Y%m%d_%H%M%S")
                ms = int((time.time() % 1) * 1000)
                fname = f"banana_{ts}_{ms:03d}_{file_index:02d}{ext}"
                fpath = out_dir_p / fname
                file_index += 1
                try:
                    with open(fpath, "wb") as f:
                        f.write(inline.data)  # already bytes
                    saved.append(str(fpath))
                    log.info("Banana saved: %s", fpath)
                    # Stop early if we hit requested count
                    if n > 0 and len(saved) >= n:
                        break
                except Exception as e:
                    return {"ok": False, "error": f"Failed to save generated image: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Generation failed: {e}"}

    return {
        "ok": True,
        "paths": saved,
        "text": "\n".join(texts).strip() if texts else "",
        "model": model,
        "count": len(saved),
        "out_dir": str(out_dir_p),
        "guided_by": input_paths,
    }


# =============================================================================
# Veo (text→video, optionally image-conditioned)
# =============================================================================

@mcp.tool()
def veo_generate_video(
    prompt: str,
    negative_prompt: str = "",
    out_dir: str = ".",
    model: str = "veo-3.1-generate-preview",
    image_path: str | None = None,   # pass a still to animate
    aspect_ratio: str | None = None, # e.g. "16:9" or "9:16"
    resolution: str | None = None,   # e.g. "720p" or "1080p" (16:9 only)
    seed: int | None = None,         # small determinism bump
    poll_seconds: int = 8,
    max_wait_seconds: int = 900,
) -> Dict[str, Any]:
    """
    Generate a video using Veo 3.1.

    - Uses the same polling pattern as your working Veo MCP tool.
    - Downloads the remote video file before saving it locally.
    """
    try:
        from google import genai
        from google.genai import types as gtypes
        import mimetypes
    except Exception as e:
        return {"ok": False, "error": f"google-genai not installed: {e}"}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"ok": False, "error": "GEMINI_API_KEY not set in environment"}

    client = genai.Client(api_key=api_key)
    out_dir_p = Path(os.path.expanduser(out_dir))
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Optional image conditioning
    image_obj = None
    if image_path:
        try:
            with open(image_path, "rb") as f:
                data = f.read()
            mt, _ = mimetypes.guess_type(image_path)
            image_obj = gtypes.Image(image_bytes=data, mime_type=mt or "image/png")
        except Exception as e:
            return {"ok": False, "error": f"read image failed: {e}"}

    cfg = gtypes.GenerateVideosConfig(
        negative_prompt=negative_prompt or None,
        aspect_ratio=aspect_ratio or None,
        resolution=resolution or None,
        seed=seed,
    )

    # Start the operation
    try:
        op = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=image_obj,
            config=cfg,
        )
    except Exception as e:
        return {"ok": False, "error": f"veo start failed: {e}"}

    # Poll until finished – IMPORTANT: use `op`, not `op.name`
    waited = 0
    try:
        while not op.done:
            if waited >= max_wait_seconds:
                return {"ok": False, "error": f"timeout after {max_wait_seconds}s"}
            time.sleep(max(1, int(poll_seconds)))
            waited += poll_seconds
            op = client.operations.get(op)
    except Exception as e:
        return {"ok": False, "error": f"veo poll failed: {e}"}

    # Grab generated videos
    # (your other project uses `operation.result.generated_videos`)
    vids = getattr(getattr(op, "result", op), "generated_videos", []) or []
    if not vids:
        return {"ok": False, "error": "no videos in response"}

    saved: list[str] = []
    for idx, gv in enumerate(vids):
        try:
            # Download the remote file so `.save()` works
            client.files.download(file=gv.video)

            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((time.time() % 1) * 1000)
            fpath = out_dir_p / f"veo_{ts}_{ms:03d}_{idx:02d}.mp4"

            gv.video.save(str(fpath))
            saved.append(str(fpath))
            log.info("Veo video saved: %s", fpath)
        except Exception as e:
            return {"ok": False, "error": f"failed to save video: {e}"}

    return {
        "ok": True,
        "paths": saved,
        "model": model,
        "seconds_waited": waited,
        "image_used": bool(image_obj),
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "seed": seed,
    }

# =============================================================================
# VibeGraphics: GitHub bundle
# =============================================================================

@mcp.tool()
def vibe_fetch_github(
    repo_url: str,
    branch: str = "main",
    include_readme: bool = True,
    include_code: bool = True,
    max_code_files: int = 10,
    max_code_chars_per_file: int = 20000,
) -> Dict[str, Any]:
    """
    VibeGraphics stage 1: Fetch GitHub content for a project.

    - repo_url: GitHub URL (HTTPS or SSH).
    - branch: usually 'main' or 'master'.
    - include_readme: if True, fetch README.* from repo root.
    - include_code: if True, fetch up to max_code_files Python files (top-level and 'src/').
    - Returns a JSON bundle path with README + code snippets for later VibeGraphic planning.
    """
    if requests is None:
        return {"ok": False, "error": "requests library not installed"}

    parsed = _normalize_github_repo(repo_url)
    if not parsed:
        return {"ok": False, "error": f"Could not parse GitHub repo from '{repo_url}'"}

    owner, repo = parsed
    log.info("VibeGraphics: fetching repo %s/%s (branch=%s)", owner, repo, branch)

    readme_text = ""
    code_files: List[Dict[str, Any]] = []

    # -------------------------
    # Try to find README.* in root
    # -------------------------
    if include_readme:
        root_items = _github_api_list_files(owner, repo, "")
        candidates = [f for f in root_items if f.get("type") == "file"]
        readme_candidates = [
            f for f in candidates
            if f.get("name", "").lower().startswith("readme")
        ]
        if readme_candidates:
            rm = readme_candidates[0]
            rm_path = rm.get("path", "README.md")
            raw_url = _github_raw_url(owner, repo, branch, rm_path)
            readme_text = _github_fetch_text_file(raw_url, max_chars=60000)
        else:
            log.info("No README found at repo root for %s/%s", owner, repo)

    # -------------------------
    # Collect Python files (top-level + src/) up to max_code_files
    # -------------------------
    if include_code and max_code_files > 0:
        code_dirs = ["", "src"]
        seen_paths: set[str] = set()

        for d in code_dirs:
            items = _github_api_list_files(owner, repo, d)
            for item in items:
                if len(code_files) >= max_code_files:
                    break
                if item.get("type") != "file":
                    continue
                name = item.get("name", "")
                if not name.endswith(".py"):
                    continue
                file_path = item.get("path", name)
                if file_path in seen_paths:
                    continue
                seen_paths.add(file_path)
                raw_url = _github_raw_url(owner, repo, branch, file_path)
                content = _github_fetch_text_file(raw_url, max_chars=max_code_chars_per_file)
                if not content:
                    continue
                code_files.append(
                    {
                        "path": file_path,
                        "chars": len(content),
                        "snippet": content,
                    }
                )
            if len(code_files) >= max_code_files:
                break

    # -------------------------
    # Build bundle + save to disk
    # -------------------------
    bundle = {
        "repo": {
            "owner": owner,
            "name": repo,
            "branch": branch,
            "url": repo_url,
        },
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "readme": readme_text,
        "code_files": code_files,
    }

    bundle_id = str(uuid.uuid4())[:8]
    bundle_path = VIBE_BASE_DIR / f"vibe_bundle_{owner}_{repo}_{bundle_id}.json"
    try:
        with open(bundle_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
    except Exception as e:
        return {"ok": False, "error": f"Failed to write bundle: {e}"}

    # Small, human-friendly hint
    short_readme = shorten(readme_text.replace("\n", " "), width=280, placeholder="...")

    return {
        "ok": True,
        "bundle_path": str(bundle_path),
        "repo": bundle["repo"],
        "readme_preview": short_readme,
        "code_file_count": len(code_files),
    }


# =============================================================================
# VibeGraphics: Plan Infographic
# =============================================================================

@mcp.tool()
def vibe_plan_infographic(
    bundle_path: str,
    theme: str = "cartographer",
    tone: str = "optimistic, exploratory, technical-but-accessible",
    model: str = "gemini-2.0-flash",
) -> Dict[str, Any]:
    """
    VibeGraphics stage 2: Turn a repo bundle into a VibeGraphic design spec.

    - bundle_path: JSON file produced by vibe_fetch_github.
    - theme: visual metaphor (e.g., 'cartographer', 'blueprint', 'cosmic').
    - tone: narrative feel for the copy.
    - model: Gemini model to use for planning.

    Returns:
      - spec_path: where the full VibeGraphic spec JSON is saved.
      - spec: the parsed JSON object.
      The spec includes:
        - projectTitle
        - oneLiner
        - sections[]
        - callouts[]
        - palette / motifs
        - imagePrompt   (for nano banana)
        - animationPrompt (for Veo)
        - voiceoverScript
    """
    if genai is None or gtypes is None:
        return {"ok": False, "error": "google-genai not installed"}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"ok": False, "error": "GEMINI_API_KEY not set in environment"}

    # -------------------------
    # Load the bundle
    # -------------------------
    try:
        with open(os.path.expanduser(bundle_path), "r", encoding="utf-8") as f:
            bundle = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"Failed to read bundle: {e}"}

    readme = bundle.get("readme", "") or ""
    code_files = bundle.get("code_files", []) or []
    repo_meta = bundle.get("repo", {}) or {}

    # Compact representation of code (paths + short snippet)
    code_summary_pieces: List[str] = []
    for cf in code_files[:8]:
        snippet = cf.get("snippet", "") or ""
        short_snippet = snippet[:1200]
        code_summary_pieces.append(
            f"### {cf.get('path')}\n```python\n{short_snippet}\n```"
        )
    code_summary = "\n\n".join(code_summary_pieces)

    # -------------------------
     # -------------------------
    # Build the instruction
    # -------------------------
    instruction = f"""
You are a world-class technical storyteller and infographic designer.

You are designing a *VibeGraphic* — an emotionally-resonant infographic that explains
a software project as if you were a **cartographer** mapping out an unexplored territory.

THEME: "{theme}"
TONE: "{tone}"

The project comes from GitHub:
- Owner: {repo_meta.get('owner', '')}
- Repo: {repo_meta.get('name', '')}
- URL: {repo_meta.get('url', '')}
- Branch: {repo_meta.get('branch', '')}

You are given:
- The full README text.
- A compact summary of some key code files (if any).

VERY IMPORTANT BEHAVIOR:

1. **Respect the README structure.**
   - If the README has headings like "Installation", "Usage", "How it works", "Features",
     or numbered steps (e.g. "1. GitHub → Bundle", "2. Bundle → Spec", "3. Spec → Image", "4. Image → Animation"),
     then REUSE those names or very close variants as section titles on the map.
   - Do NOT replace everything with generic names like "The Treasure" or "Exploration" unless they
     are clearly aligned with a specific README section.

2. **Create a PIPELINE section that mirrors the README steps.**
   - If the README describes a process or pipeline (for example:
        1. Fetch repo
        2. Plan infographic
        3. Render image
        4. Animate with Veo
     ), then include a section that visually shows these steps in order.
   - Use the SAME wording and ordering as the README wherever possible.
   - This section should look like a connected route or path on the map with labels like
     "Step 1: GitHub → Bundle", "Step 2: Bundle → Spec", etc., if the README contains those ideas.

3. **Keep terminology project-specific.**
   - Preserve important proper nouns from the README: project name ("VibeGraphics"), tool names
     ("nano banana", "Veo", "Gemini", "MCP", "Gemini CLI", etc.).
   - Do not replace them with vague words like "our system", "the tool", etc.

4. **Make the graphic strongly tied to the README.**
   - Every section in the infographic should be clearly traceable back to some part of the README
     (a heading, a paragraph, or a list).
   - Use the README content to fill in the copy and labels first; only add new wording if needed
     to make it flow visually.

Return STRICT JSON with this structure and ONLY this structure:

{{
  "projectTitle": "Short human-friendly title for the project",
  "oneLiner": "A single-sentence explanation of the project in plain language",
  "sections": [
    {{
      "id": "overview",
      "title": "Section title (prefer actual README heading or step name)",
      "body": "2–5 short sentences of copy for this section, closely based on the README text",
      "iconIdea": "Visual motif in the cartographer theme (e.g., compass, map grid, waypoint)",
      "layoutHints": "Where this might sit on the infographic (e.g. top-left, center, bottom strip)"
    }}
  ],
  "callouts": [
    {{
      "label": "A very short label, like a map legend item, ideally pulled from README phrasing",
      "text": "One short sentence of supporting info based on the README",
      "visualMarker": "Small visual symbol, like 'dotted path', 'mountain peak', etc."
    }}
  ],
  "palette": {{
    "primaryColors": ["#hex", "#hex"],
    "accentColors": ["#hex", "#hex"],
    "backgroundStyle": "Brief description of background (e.g. aged parchment map, night-sky star chart)",
    "typographyStyle": "Description of font feel (e.g. clean sans-serif with subtle serif headings)"
  }},
  "imagePrompt": "A detailed text-to-image prompt for generating the static infographic in the chosen theme. It MUST clearly mention any important README sections or steps by name, and describe how they are placed on the map.",
  "animationPrompt": "A detailed prompt for animating this infographic: describe subtle movements, camera motions, transitions between sections, and how to highlight key elements. The animation should follow the same order as the README's main sections or steps.",
  "voiceoverScript": "A concise voiceover script (60–90 seconds) that walks through the infographic in the same order as the README's main sections or numbered steps."
}}

Constraints:
- Prefer using exact section names and step names from the README where possible.
- The imagePrompt should be directly usable by a model like nano banana.
- The animationPrompt should be directly usable by a video model like Veo 3, assuming we pass the static infographic as the starting image.
- Keep JSON reasonably compact; avoid huge blockquotes or long code samples.
"""


    client = genai.Client(api_key=api_key)

    parts: List[gtypes.Part] = [
        gtypes.Part.from_text(text=instruction),
        gtypes.Part.from_text(text="### README\n" + readme[:20000]),
    ]
    if code_summary:
        parts.append(gtypes.Part.from_text(text="\n\n### CODE SUMMARY\n" + code_summary))

    content = gtypes.Content(role="user", parts=parts)

    try:
        res = client.models.generate_content(
            model=model,
            contents=[content],
            config=gtypes.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        raw = getattr(res, "text", "") or "{}"
        spec = json.loads(raw)
    except Exception as e:
        return {
            "ok": False,
            "error": f"Gemini planning failed: {e}",
            "raw": raw if "raw" in locals() else "",
        }

    # -------------------------
    # Persist spec to disk
    # -------------------------
    spec_id = str(uuid.uuid4())[:8]
    spec_path = VIBE_BASE_DIR / f"vibe_spec_{repo_meta.get('name','project')}_{spec_id}.json"
    try:
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
    except Exception as e:
        return {"ok": False, "error": f"Failed to write VibeGraphic spec: {e}"}

    return {
        "ok": True,
        "spec_path": str(spec_path),
        "spec": spec,
        "repo": repo_meta,
        "model": model,
    }


# =============================================================================
# VibeGraphics: Convenience wrappers (spec -> image, spec -> animation)
# =============================================================================

@mcp.tool()
def vibe_render_image_from_spec(
    spec_path: str,
    out_dir: str = ".",
    model: str = "gemini-2.5-flash-image-preview",
    n: int = 1,
) -> Dict[str, Any]:
    """
    Given a VibeGraphic spec JSON (from vibe_plan_infographic),
    generate the static infographic image using banana_generate().

    - spec_path: path to VibeGraphic spec.
    - out_dir: where to write the generated image(s).
    - model: Gemini image generation model.
    - n: desired number of images.

    Uses spec["imagePrompt"] as the prompt.
    """
    try:
        with open(os.path.expanduser(spec_path), "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"Failed to read spec: {e}"}

    image_prompt = spec.get("imagePrompt") or ""
    if not image_prompt:
        return {"ok": False, "error": "Spec missing 'imagePrompt'"}

    # Call the internal banana_generate helper
    result = banana_generate(
        prompt=image_prompt,
        input_paths=None,
        out_dir=out_dir,
        model=model,
        n=n,
    )

    if not result.get("ok", False):
        return {
            "ok": False,
            "error": f"banana_generate failed: {result.get('error','unknown error')}",
        }

    return {
        "ok": True,
        "paths": result.get("paths", []),
        "model": model,
        "spec_path": spec_path,
    }


@mcp.tool()
def vibe_animate_from_spec(
    spec_path: str,
    image_path: str,
    out_dir: str = ".",
    model: str = "veo-3.1-generate-preview",
    aspect_ratio: Optional[str] = None,
    resolution: Optional[str] = None,
    seed: Optional[int] = None,
    lock_camera: bool = True,
) -> Dict[str, Any]:
    """
    Given a VibeGraphic spec JSON and a static infographic image,
    generate an animated VibeGraphic using Veo.

    - spec_path: path to VibeGraphic spec.
    - image_path: path to the static infographic (e.g. output of vibe_render_image_from_spec / banana_generate).
    - out_dir: directory for the generated video(s).
    - model: Veo model to use.
    - aspect_ratio: optional aspect ratio, e.g. "16:9" or "9:16".
    - resolution: optional resolution, e.g. "720p" or "1080p".
    - seed: optional seed for slight determinism.
    - lock_camera: if True, keep the full poster in frame and only animate
      elements inside it (no big zooms/pans or scene cuts).
    """
    try:
        with open(os.path.expanduser(spec_path), "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"Failed to read spec: {e}"}

    base_prompt = spec.get("animationPrompt") or ""
    if not base_prompt:
        return {"ok": False, "error": "Spec missing 'animationPrompt'"}

    # --- IMPORTANT: constrain Veo to treat this as a flat poster, not a movie ----
    if lock_camera:
        constraints = (
            "\n\nANIMATION CONSTRAINTS (IMPORTANT):\n"
            "- the idea in not to make a movie, but to animate a single infographic poster.\n"
            "- Treat the supplied image as a single flat poster / infographic.\n"
            "- Keep the entire poster visible in frame for the whole clip.\n"
            "- Do NOT cut to new scenes or generate new backgrounds.\n"
            "- Avoid large camera moves: no hard pans, no big zoom-ins or zoom-outs.\n"
            "- If you move the camera at all, keep it extremely subtle "
            "(gentle 5–10% drift or parallax only).\n"
            "- All motion must happen INSIDE the existing elements of the poster:\n"
            "    * soft glow pulses\n"
            "    * icons gently bobbing or rotating\n"
            "    * lines drawing themselves in\n"
            "    * small sparkles or highlights on key areas\n"
            "- Never crop away edges of the infographic; the user should always see the whole design.\n"
        )
        animation_prompt = base_prompt + constraints
    else:
        animation_prompt = base_prompt

    # Call the internal veo_generate_video helper
    result = veo_generate_video(
        prompt=animation_prompt,
        negative_prompt="",
        out_dir=out_dir,
        model=model,
        image_path=image_path,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        seed=seed,
    )

    if not result.get("ok", False):
        return {
            "ok": False,
            "error": f"veo_generate_video failed: {result.get('error','unknown error')}",
        }

    return {
        "ok": True,
        "paths": result.get("paths", []),
        "model": model,
        "spec_path": spec_path,
        "image_path": image_path,
        "lock_camera": lock_camera,
    }



# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
