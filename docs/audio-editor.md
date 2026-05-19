# Audio Editor

Post-processing tools available in the Gradio Edit tab.

The Gradio UI includes a built-in **Edit** tab for quick post-processing without
leaving Anvil. It works with any loaded model and any audio file — not just
files generated in the current session.

Available tools:

| Tool | What it does |
|---|---|
| Normalize | Peak or LUFS loudness targeting |
| Trim silence | Strips quiet sections from the edges |
| Fade in / fade out | Linear ramp from/to silence |
| Loop / clip | Trim the audio to a start/end range |
| Time stretch | Speed up or slow down without changing pitch |
| Pitch shift | Transpose up or down in semitones |
| EQ | Low shelf, peak mid, high shelf |
| Reverb | Room size, damping, wet/dry mix |

Effects are applied in a fixed chain (trim → clip → stretch/pitch → EQ →
reverb → fade → normalize), which keeps results predictable regardless of the
order you adjust knobs.

**Typical workflow:**

1. Generate on the **Generate** tab
2. Switch to **Edit**
3. Click **Load Last Generation** — the output loads automatically
4. Adjust effects; click **Preview** to hear the result
5. Click **Export** when satisfied

Export creates a new file via the output manager — the original is never
touched. The JSON sidecar for the exported file records the source path and
the full effects chain so you can always trace what was applied and replay it.

You can also drag any audio file into the source field to edit files from
outside Anvil.

---
