# Third-party notices and provenance

## ReefVision theme reference

ReefVision-Edge's semantic colors and related visual conventions were derived from Rafael Del Valle's ReefVision at commit `e94dad6a3bece7b776bf45776bc5d06cfbc2c0dc`, especially:

- `apps/web/src/main.tsx`
- `apps/web/src/App.tsx`
- `apps/web/src/styles/clarity-overhaul.css`
- `apps/web/src/styles/design-system.css`
- `apps/web/src/styles/app.css`

ReefVision is licensed under the GNU Affero General Public License v3.0 only. ReefVision-Edge is distributed under the same license. Copyright and license terms remain with their respective owners. The full license is in `LICENSE`.

No remote fonts, icons, model weights, or image assets from ReefVision are bundled. The Edge wordmark, CSS layout, and synthetic test imagery are local to this repository.

## Runtime and build dependencies

Python and JavaScript dependencies retain their own licenses. Important runtime components include FastAPI, Starlette, Uvicorn, Pydantic, NumPy, OpenCV, ONNX Runtime, React, and Vite. Consult the installed package metadata and lock file for exact versions and license texts.

The Windows CUDA runtime profile redistributes NVIDIA CUDA 13.4 libraries (CUDA Runtime, NVRTC, nvJitLink, cuBLAS, cuFFT, and cuRAND) and NVIDIA cuDNN 9 under their respective NVIDIA license terms. Exact versions and wheel SHA-256 values are recorded in `packaging/windows/runtime-windows-cuda.lock`. These libraries accelerate ONNX Runtime inference only; the required monitor-video baseline remains the OpenCV software encoder.
