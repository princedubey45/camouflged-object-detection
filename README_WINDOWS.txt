ESCNET WINDOWS NVIDIA SETUP

1. Copy the full Dubey folder to the Windows laptop.
2. Keep this ESCNet folder at:
   Dubey\cod_project\ESCNet
3. Install Python 3.11 on the Windows laptop.
4. Open Command Prompt in this ESCNet folder.
5. Run:

setup_cuda_windows.bat

6. Start training with:

train_windows.bat

7. Run video inference with:

python video_infer.py --input path\to\input_video.mp4 --ckpt checkpoints\escnet\epoch_80.pth --output outputs\video_prediction.mp4

NOTES

- This uses the ESCNet repo-style training command.
- The current config keeps epochs at 80 and image size at 340.
- The config expects:
  ..\COD10K-v3\Train
  ..\COD10K-v3\Test
  ..\..\pvt_v2_b5.pth
- If CUDA is installed correctly, training will use the NVIDIA GPU.
