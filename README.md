# EarthScope AI

A futuristic Earth-observation image analysis prototype.

## What it does

1. Uploads an image.
2. Checks whether it is readable and sufficiently sharp.
3. Uses CLIP to determine whether the image is relevant to Earth/space observation.
4. Uses CLIPSeg to look for supported visual features:
   - clouds
   - ocean / water
   - forest / vegetation
   - urban area
   - ice / snow
   - desert
   - mountains
   - fire / smoke
5. Produces highlighted regions, feature confidence, and a simple-language explanation.
6. Has three user-facing outcomes:
   - successful analysis
   - blurry/unrelated image
   - system error
7. If no feature reaches the confidence threshold, it returns a valid "no supported features detected" result rather than inventing one.

## Important

This is a working prototype, not a scientific remote-sensing instrument. CLIP/CLIPSeg are general vision models and should not be treated as authoritative scientific classification.

## Requirements

- Python 3.10 or 3.11 recommended
- Internet connection on first run because Hugging Face model weights are downloaded
- About several GB of free disk/RAM depending on the model/runtime

## Windows setup

Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

Then open:

http://127.0.0.1:5000

## CPU/GPU

The app automatically uses CUDA when PyTorch detects an NVIDIA GPU; otherwise it uses CPU.

The first analysis can be slow because the models are downloaded and loaded. Later analyses are faster.

## Testing the three outcomes

### Successful result
Use a clear satellite/aerial Earth image containing clouds, water, vegetation, etc.

### Blurry result
Upload a very small or deliberately blurred image.

### Unrelated result
Upload a normal unrelated photograph, such as a person or indoor object.

### System error
For development only, you can add `?force_error=1` to the POST request using a browser dev tool or modify the `FORCE_ERROR_FOR_DEMO` setting in `app.py`.

## Project structure

```text
earthscope_ai/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── app.js
```

## Replacing the AI later

The backend keeps the AI work inside these functions:

- `check_image_quality`
- `check_relevance`
- `detect_features`
- `build_explanation`

That means a stronger remote-sensing model can replace the current CLIP/CLIPSeg implementation without redesigning the frontend.
