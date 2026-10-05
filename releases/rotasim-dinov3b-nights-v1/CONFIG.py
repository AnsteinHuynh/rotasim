"""User configuration for the SinkOFTsim package (edit this file).

The released checkpoint contains ONLY the trained OFT adapters (~1.7 MB).
The frozen DINOv3-B/16 backbone weights must be downloaded separately,
because Meta keeps DINOv3 behind a gated Hugging Face repo (you must be
logged in and have accepted the license once).

HOW TO GET THE BACKBONE (one-time)
----------------------------------
1. Create a Hugging Face account, log in, and accept the license at
   https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
2. Download it, e.g.:

       pip install -U "huggingface_hub[cli]"
       huggingface-cli login
       huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m ^
           --local-dir C:/myPath/dinov3-vitb16-pretrain-lvd1689m

   (on Linux/macOS drop the ^ line continuation)

HOW THE PATH WORKS
------------------
The downloaded folder is a standard HF snapshot and looks like:

    C:/myPath/dinov3-vitb16-pretrain-lvd1689m/
        config.json
        model.safetensors
        preprocessor_config.json
        ... (a few .json / .txt sidecar files)

Point the variable below at that FOLDER (any location works) and the
package will load the tower from there.

If you leave a variable EMPTY (""), the package falls back to pulling
the model straight from the Hugging Face Hub by repo id -- that works
only if you ran `huggingface-cli login` on this machine and have
accepted the model's gate in the browser.
"""

# ---------------------------------------------------------------------------
# Local folder containing the backbone snapshot. Example:
#
#     DINOV3_VITB16_PATH = r"C:/myPath/dinov3-vitb16-pretrain-lvd1689m"
#
# This folder should contain model.safetensors, config.json,
# preprocessor_config.json, etc. (exactly what huggingface-cli download
# produces). Leave "" to fetch from the Hub automatically.
# ---------------------------------------------------------------------------
DINOV3_VITB16_PATH = ""

# Optional towers used by other checkpoints in this family (not needed for
# the NIGHTS-544 release, which is DINOv3-B/16 only):
DINOV3_VITL16_PATH = ""      # fallback: facebook/dinov3-vitl16-pretrain-lvd1689m
SIGLIP2_BASE16_PATH = ""     # fallback: google/siglip2-base-patch16-224
