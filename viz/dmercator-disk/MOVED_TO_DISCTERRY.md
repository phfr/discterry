# This folder was renamed

The WebGPU disk viewer now lives at **`../discterry/`** (package name **discterry**).

Stop any dev server using this path, then delete this `dmercator-disk` directory if it is still present.

```bash
cd viz/discterry
npm install
npm run dev
```

Export path in `notebooks/dmercator/04_export_disk_web_bundle.ipynb` writes to `viz/discterry/public/data/`.
