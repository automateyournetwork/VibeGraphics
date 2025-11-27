# VibeGraphics
An MCP Server / Gemini CLI Extension to create modern Infographics for your project - VibeGraphics - Generated with Nano Banana and animated with Veo3

## Installation

gemini extensions install https://github.com/automateyournetwork/VibeGraphics.git

📜 VibeGraphics
AI-Generated Infographics & Animations From Your GitHub Projects

VibeGraphics is an AI-powered toolkit that transforms any GitHub project into a beautiful, theme-based infographic — and optionally a short animated video — using multimodal models like Gemini, nano banana (image generation), and Veo (video).

Provide a GitHub URL → receive a fully generated “VibeGraphic” that visually explains your project’s purpose, architecture, components, and flow.

🌟 Key Features
🔍 1. Project Scraping (GitHub → Bundle)

VibeGraphics fetches:

README

Source code snippets

File structure

Metadata (owner, repo, branch)

…and packages them into a compact analysis bundle.

🎨 2. Infographic Design (Bundle → Spec)

VibeGraphics uses large language models to create a VibeGraphic Spec, including:

Title & one-liner

Sections & descriptions

Visual motifs (e.g., cartographer, cosmic, blueprint)

Color palette

Layout hints

Image prompt (for rendering)

Animation prompt (for Veo)

Optional voiceover script (60–90s narration)

This is a design document describing the infographic.

🖼 3. Image Generation (Spec → Infographic)

Using the spec’s imagePrompt, VibeGraphics creates a static, high-quality infographic using:

nano banana (Gemini image generation)

Optional guided mode using input images

The result is a single, visually consistent graphic representing your GitHub project.

🎬 4. Animation (Image → Motion Graphic)

With Veo, VibeGraphics can animate the infographic with:

Parallax motion

Section reveals

Camera glides

Compass moves, highlights, sparkles (theme-dependent)

Produces a 5–12 second animated motion graphic, suitable for:

Project landing pages

Social posts

Presentations

Documentation headers

🚀 How It Works (High-Level)

Provide a GitHub URL.

VibeGraphics scrapes and analyzes the repository.

A VibeGraphic Spec is generated using multimodal AI.

An infographic is rendered using the spec’s image prompt.

Optional: the infographic is animated using Veo.

You receive:

Spec JSON

Infographic image

Animated video (optional)

📦 Project Structure (High-Level Overview)
vibegraphics/
├── vibegraphics_mcp.py   # MCP server: GitHub fetch → Spec → Image → Animation
├── servers/
│   ├── requirements.txt
│   └── run.sh
├── extensions/
│   ├── GEMINI.md         # LLM-facing instructions
│   └── commands.toml     # Slash commands for Gemini-CLI
└── README.md             # (this file)

🔧 Installation
pip install google-genai fastmcp requests
export GEMINI_API_KEY="YOUR_KEY_HERE"


Or when used as a Gemini-CLI extension:

gemini extensions install .

🧠 Example Usage (Conceptual)
Generate an infographic of a GitHub repo:
Make a vibe graphic for https://github.com/myuser/myproject

Custom theme:
Create a cosmic-style vibegraphic of this repo:
https://github.com/myuser/myproject

Full pipeline:
Turn this repo into a vibegraphic and animate it:
https://github.com/myuser/myproject

🗺 Themes (Current & Planned)

Current default:

Cartographer – parchment maps, compass rose, routes, topographic lines

Planned:

Cosmic Starfield

Blueprint

Retro Terminal

Futuristic Neon

Botanical

Minimalist Diagram

Architectural Drafting

🎯 Why VibeGraphics?

Software is complicated.
Documentation is overwhelming.
Most repos deserve something beautiful that captures the vibe of the project.

VibeGraphics:

Helps developers understand your project at a glance

Creates shareable visuals for socials, docs, and presentations

Turns abstract code into emotional, intuitive visuals

Feels like branding for your GitHub project

🤝 Contributing

Contributions, themes, and prompt enhancements are welcome.
Feel free to open:

Issues

Pull requests

Theme suggestions

Prompt design ideas

New animation patterns (e.g., neon flicker, cosmological drift)

📜 License

MIT License – use, remix, adapt, and build your own VibeGraphics pipelines.

Ready to Create Your First VibeGraphic?

Just point VibeGraphics to a repo and let the generative design engine do the rest.

If you need a VibeGraphic of this VibeGraphics repo, just ask:

“Create a vibe graphic for this project.”