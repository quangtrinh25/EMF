| Thuộc tính | Bài báo | v3.2.2 selected | v4.4 research |
|---|---|---|---|
| Input | 9 | W3 raw: 27 hoặc +2 unit-row | 29 packed → 47 internal |
| Backbone | 512, 7 residual blocks | Mỗi member: 512, 7 blocks | 512, 7 blocks |
| Block | 2×(512 linear), LeakyReLU 0.01 | Giống | Giống |
| Head/output | 512→128→6 | 512→128→9 | 512→128→9 |
| Orientation | cos Euler | rotation-6D | rotation-6D |
| Temporal | Không | Causal W3 | Causal W3 + dt_ratio |
| Calibration adapter | Không | Không | Per-generation 9 gain + 9 bias |
| Feature | Raw EMF | Raw hoặc log-ratio branch | 27 raw + 18 log-ratio + 2 dt |
| Composition | 1 model | 5-member component ensemble | Frozen v3.2.2 + C3 weighted blend |
| Status | Published proposed | Selected deployment | Research-only; not deployed |
