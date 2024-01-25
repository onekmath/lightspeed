# Light Speed ⚡

Light Speed ⚡ is an open-source text-to-speech model based on VITS, with some modifications:
- utilizes phoneme duration's ground truth, obtained from an external forced aligner (such as Montreal Forced Aligner), to upsample phoneme information to frame-level information. The result is a more robust model, with a slight trade-off in speech quality.
- employs dilated convolution to expand the Wavenet Flow module's receptive field, enhancing its ability to capture long-range interactions.

<!-- ![network diagram](net.svg) -->

## Step to create tfdata
1. create folder data/mydata.
2. run create_lexicon.py to create lexicon.txt.
3. run create_mfa.bat to create mfa model json.
4. create tfdata.
5. use tfdata to train vits model and duration model
