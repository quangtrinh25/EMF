| Split | Scenario | Method | Position RMSE (mm) | Direction RMSE Eq.9 (deg) | SO(3) RMSE (deg) | Position p95 (mm) | SO(3) p95 (deg) | CPU p95 (ms) | GPU p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Development CV | con_rot s1 | v3.2.2 | 0.9890 | 0.2236 | 0.3872 | 2.8701 | 0.6972 | 1.2902 | 0.8581 |
| Development CV | con_rot s1 | Hybrid w=0.20 | 0.9249 | 0.2128 | 0.3685 | 2.8110 | 0.6520 | 1.9923 | 1.4226 |
| Development CV | con_rot s2 | v3.2.2 | 0.5753 | 0.3826 | 0.6628 | 1.7621 | 0.9896 | 1.2902 | 0.8581 |
| Development CV | con_rot s2 | Hybrid w=0.20 | 0.6351 | 0.2644 | 0.4579 | 1.8099 | 0.7333 | 1.9923 | 1.4226 |
| Development CV | con_rot s3 | v3.2.2 | 0.6050 | 0.1245 | 0.2157 | 1.5680 | 0.2894 | 1.2902 | 0.8581 |
| Development CV | con_rot s3 | Hybrid w=0.20 | 0.5927 | 0.1160 | 0.2009 | 1.5438 | 0.2835 | 1.9923 | 1.4226 |
| One-time final | cyl_rot pooled | v3.2.2 (selected) | 1.1222 | 1.3801 | 2.3904 | 3.1047 | 5.0578 | 1.2902 | 0.8581 |
| One-time final | cyl_rot pooled | Hybrid w=0.20 (rejected) | 1.2335 | 1.0769 | 1.8653 | 3.4725 | 3.7461 | 1.9923 | 1.4226 |
