# Dual-pol CycleGAN: SAR -> optical (Earth proxy for lunar PSRs)

Unpaired image-to-image translation. Domain A = Sentinel-1 (VV, VH), domain B = Sentinel-2 optical.
Trained on the Bengaluru proxy; designed so the same code can later take lunar SAR/optical tiles.

## Files
| file | purpose |
|---|---|
| `prepare_data.py` | S1/S2 rasters -> normalised 256px tiles, Lee speckle filter, spatial train/val/test split |
| `prepare_lunar_data.py` | Lunar Mini-RF/DFSAR + LROC NAC/ShadowCam -> normalized tiles with PSR mask filtering |
| `make_synthetic.py` | fake Earth proxy scene for smoke-testing the pipeline (not real data) |
| `make_lunar_synthetic.py` | fake lunar scene with impact craters, PSR crater bowls & ShadowCam for testing |
| `dataset.py` | unpaired training dataset + paired evaluation dataset |
| `models.py` | ResNet generator, 70x70 PatchGAN discriminator |
| `losses.py` | LSGAN loss, image pool, optional edge-consistency prior |
| `train.py` | CycleGAN training (GAN + cycle-consistency) |
| `evaluate.py` | SSIM / PSNR / correlation on the held-out test split vs. a mean-image baseline |

## 1. Get the data (Google Earth Engine, JavaScript editor)
Both rasters must be on the **same grid**, so export them with identical region, CRS and scale.
```js
var region = ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.10]);   // Bengaluru, adjust
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(region).filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))   // keep one geometry
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']).median();                                // already in dB
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region).filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .select(['B4', 'B3', 'B2']).median();                          // reflectance x 10000
var opts = {region: region, scale: 10, crs: 'EPSG:32643', maxPixels: 1e10};
Export.image.toDrive(Object.assign({image: s1.toFloat(), description: 'blr_s1'}, opts));
Export.image.toDrive(Object.assign({image: s2.toFloat(), description: 'blr_s2'}, opts));
```
Use temporal medians so clouds and speckle are reduced; keep `--s1-units db`.

## 2. Prepare tiles
```bash
python prepare_data.py --s1 blr_s1.tif --s2 blr_s2.tif --out data/blr --optical-mode gray
```
`gray` makes the optical target a single luminance channel, closer to lunar optical imagery.
Use `rgb` if you want colour output.

## 3. Train
```bash
python train.py --data data/blr --channels vv,vh --out runs/dualpol          # proposed
python train.py --data data/blr --channels vv    --out runs/vv_only          # ablation
python train.py --data data/blr --channels vh    --out runs/vh_only          # ablation
python train.py --data data/blr --channels vv,vh,ratio --out runs/dualpol_ratio
```
Default is 100 + 100 decay epochs. Use a GPU. `samples/epoch_XXX.png` columns are:
SAR (first channel) | generated optical | real optical | cycle-reconstructed SAR.

## 4. Evaluate
```bash
python evaluate.py --ckpt runs/dualpol/last.pt --data data/blr
```
The test split is a separate band of the scene, so there is no tile overlap with training.
Training stays unpaired, but the Earth S1/S2 tiles are co-registered, so test scoring against the true optical tile is legitimate.
Compare dual-pol vs single-pol runs for your core claim. Watch `corr` and `ssim` more than `psnr_db`.

## Smoke test (no data needed)
```bash
python make_synthetic.py --out raw_synthetic
python prepare_data.py --s1 raw_synthetic/s1.npy --s2 raw_synthetic/s2.npy --out data/synth
python train.py --data data/synth --epochs 1 --epochs_decay 1 --n_blocks 3 --ngf 16 --out runs/smoke
python evaluate.py --ckpt runs/smoke/last.pt --data data/synth
```

## Moving to the Moon (later)
- Linear VV/VH does not exist in lunar data: Mini-RF/DFSAR are circular/hybrid-pol. Map your two channels to same-sense/opposite-sense (or S1/S2 Stokes) products, and retrain rather than reusing Earth weights.
- Target domain: LROC NAC (sunlit terrain) as grayscale optical. ShadowCam imagery gives a qualitative check inside PSRs.
- Sunlit regions have both radar and LROC coverage, so they give paired lunar validation data, which is the strongest evidence you can show.
- Outputs inside PSRs are plausible renderings, not measurements. Say so in the write-up.
