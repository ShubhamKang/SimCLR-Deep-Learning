"""Convert the public SimCLR STL-10 checkpoint (sthalles/SimCLR, MIT licence) to our format.

Source: https://github.com/sthalles/SimCLR  (README table: ResNet-18, STL-10, 100 epochs, 74.45% linear probe)
Download: https://drive.google.com/open?id=14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF  (a zip with checkpoint_0100.pth.tar)

Why this is needed: their model stores the ResNet-18 under the prefix "backbone." and puts the
projection head in "backbone.fc". Our encoder is a plain torchvision ResNet-18 with fc removed
(src/model.py), so we strip the prefix and drop the head. The result loads with evaluate.py.

Note: their training used images WITHOUT mean/std normalisation, so evaluate this checkpoint
with  --no_normalize  (otherwise the inputs differ from what the encoder saw during training).

Usage:
  python scripts/convert_public_checkpoint.py path/to/checkpoint_0100.pth.tar checkpoints/public_sthalles.pt
"""
# sys gives access to the command-line words (sys.argv); torch is PyTorch.
import sys
import torch
# Let Python find our src/ folder when this script is run from the project folder.
sys.path.insert(0, ".")
# Our ResNet-18 encoder definition (src/model.py), used to check the converted weights fit.
from src.model import make_encoder

# The two words typed after the script name: the input file (src) and the output file (dst).
src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)  # the file also stores non-tensor metadata
# The weights themselves are stored under the key "state_dict": a dictionary that maps each
# weight's name (e.g. "backbone.conv1.weight") to its numbers.
state = ckpt["state_dict"]

# Keep only the backbone, without the projection head, and remove the "backbone." prefix.
enc_state = {k[len("backbone."):]: v for k, v in state.items()
             if k.startswith("backbone.") and not k.startswith("backbone.fc.")}

# k[len("backbone."):] cuts the "backbone." prefix off each name, so the names match ours.
# Create an empty encoder with our architecture, then copy the weights in.
# strict=True checks every weight of our ResNet-18 is present and nothing is left over.
enc = make_encoder()
enc.load_state_dict(enc_state, strict=True)
# Save in the same format as train_simclr.py ({"encoder": ...}), so evaluate.py can load it.
torch.save({"encoder": enc.state_dict(), "args": {"source": "sthalles/SimCLR checkpoint_0100 (STL-10, 100 epochs)"},
            "epoch": ckpt.get("epoch", 100)}, dst)
# Report how many weight tensors were converted.
print(f"Saved {len(enc_state)} tensors to {dst}")
