# Báo cáo đối chiếu toàn bộ pipeline với bài báo tham chiếu

Ngày lập: 2026-08-04; cập nhật canonical audit: 2026-08-05  
Phạm vi: định vị 6-DoF viên nang từ chín kênh EMF trong workspace
`100 × 100 × 100 mm`  
Bài báo tham chiếu: [Residual Neural Network for Precise 6-DoF Capsule
Endoscope Localization Using Electromagnetic
Induction](../Residual_Neural_Network_for_Precise_6-DoF_Capsule_Endoscope_Localization_Using_Electromagnetic_Induction.pdf),
DOI `10.1109/TIM.2026.3655917`.

> Báo cáo canonical mới, tự dựng lại 13 hình và 6 bảng với manifest SHA-256:
> [reports/full_pipeline_comparison_2026_08_05/REPORT.md](../reports/full_pipeline_comparison_2026_08_05/REPORT.md).
> Báo cáo mới tách rõ v3.2.2 selected, mixed v4.2 final-rejected và v4.4
> synthetic→real research-only; file hiện tại được giữ như phần giải thích chi
> tiết/lịch sử, không phải nguồn quyết định deployment mới nhất.

## 1. Kết luận trung thực

Pipeline hiện tại **không phải là bản tái tạo đầy đủ toàn bộ bài báo**. Ta đã
tái tạo đúng bài toán lõi `9 EMF → pose 6-DoF`, dùng họ mạng residual rộng 512
với bảy residual block, huấn luyện và đánh giá trên dữ liệu robot thật, rồi bổ
sung nhiều cơ chế mà bài báo không có: cửa sổ thời gian nhân quả, rotation-6D,
grouped cross-validation theo session, test khóa một lần, ensemble riêng cho
vị trí/hướng, calibration theo fold, synthetic data có quality gate, CPU/GPU
runtime và giao diện predictor trạng thái.

Tuy nhiên, bốn giới hạn ngăn ta tuyên bố “đã tái tạo hoặc vượt bài báo” là:

1. Bài báo dùng workspace `500 mm`; ta dùng `100 mm` và quỹ đạo khác.
2. Bài báo kiểm tra roll, pitch, yaw thay đổi rộng `20°–160°`; final test của ta
   gần như chỉ thay đổi pitch `10°`, trong khi roll chỉ đổi `0.02°` và yaw
   `0.04°`.
3. Ta chưa huấn luyện FCN, KAN và nonlinear optimization trên cùng dữ liệu hiện
   tại, nên chưa có benchmark ngang hàng với Table II của bài báo.
4. Physics-consistency hiện không đạt: EMF reconstruction MAPE trung bình vẫn
   `106.19%` ngay cả khi dùng ground-truth pose. Bài báo báo cáo tất cả kênh
   dưới `3.4%`.

Mô hình được chọn để triển khai vẫn là `v3.2.2`. Mixed candidate v4.2 đã bị
quality gate final loại vì làm sai số vị trí tăng. Candidate v4.4
synthetic→real mới chỉ qua accuracy/session gate trên development; nó chưa có
final hợp lệ, fail GPU latency gate và được giữ research-only.

## 2. Hai pipeline đặt cạnh nhau

| Giai đoạn | Bài báo | Pipeline hiện tại | Trạng thái/ý nghĩa |
|---|---|---|---|
| Bài toán vật lý | 3 TX × 3 RX tạo chín biên độ EMF; dự đoán pose 6-DoF | Cùng dạng đầu vào/chỉ tiêu | Đã bám sát ở mức bài toán |
| Workspace | `500 × 500 × 500 mm` | `100 × 100 × 100 mm`, tâm xấp xỉ `[-99.02, 390.255, 270.88] mm` | Khác điều kiện; không so sánh trực tiếp RMSE tuyệt đối |
| Dữ liệu mô phỏng | Lưới `40³` vị trí × `10³` hướng = 64 triệu pose | v3.2.2 không dùng synthetic; v4 dùng synthetic bằng số hàng real theo tỷ lệ 1:1 | v4 ít synthetic hơn rất nhiều và đã bị loại ở final |
| Dữ liệu calibration thật | 150 pose ngẫu nhiên để fit tham số forward model | 6.019 row thuộc chín session non-test; fit physics riêng trong từng fold và một fit deployment | Nhiều hàng hơn nhưng quỹ đạo có cấu trúc và orientation hẹp |
| Chia dữ liệu | Mô phỏng chia 70/15/15; real benchmark 100 pose | Giữ nguyên session khi chia; ba fold leave-one-session-out; `cyl_rot` khóa cho final | Chống lẫn hàng/cửa sổ tốt hơn, nhưng quỹ đạo giữa session còn giống nhau |
| Input mỗi dự đoán | Một row gồm chín EMF | Cửa sổ W3 nhân quả; raw, past/current log-ratio và hai delta-row | Ý tưởng mới so với bài báo |
| Thời gian | Không dùng | Dữ liệu hiện không có hardware timestamp; mỗi row là một bước thời gian lập trình | Không được gọi delta-row là timestamp thật |
| Chuẩn hóa | Gaussian standardization | Thống kê chỉ fit từ train/fold train | Bám sát nguyên tắc, chặt hơn về split |
| Mạng | Một FCN residual, width 512, 7 block, LeakyReLU | Năm residual model trong v3.2.2; candidate v4 C3 cũng width 512, 7 block | Kiến trúc lõi bám sát, quy tắc ensemble là mới |
| Hướng | `cos(roll,pitch,yaw)` | Rotation-6D, chiếu Gram–Schmidt về SO(3), đánh giá geodesic | Mới; tránh mất dấu và xử lý hình học quay đầy đủ hơn |
| Calibration | Fit vị trí/hướng/số vòng TX rồi retrain | Fit physics chỉ trên fold train; channel gain/bias; adapter theo generation; pretrain synthetic rồi fine-tune real | Mới về kiểm soát leakage, nhưng physics chưa generalize sang final rotation |
| Sampling | Không báo cáo cân bằng theo không gian/session | C3 có sampler cân bằng session và ô không gian `5 × 5 × 5`, trọng số bị chặn | Đã dùng trong candidate v4, không có trong deployment v3.2.2 |
| Baseline | Optimization, FCN, KAN trên cùng real dataset | Chưa chạy ba baseline này trên current data | Khoảng trống bắt buộc phải công khai |
| Chọn mô hình | Dựa trên validation/test mô tả trong bài | Bốn metric, composite ≥3%, giới hạn regression theo session, latency gate | Quality gate nhiều mục tiêu là mới |
| Realtime | Báo cáo `0.82 ms` trên RTX 3090/i9 | v3.2.2 inference-core: GPU p95 `0.858 ms`, CPU p95 `1.290 ms` | Khác hardware, statistic và phạm vi đo; không tuyên bố nhanh hơn bài báo |
| Interface | Hướng tới edge-AI | JSONL stateful nhận timestamp/reset/EMF, CPU fallback và CUDA Graph | Predictor đã có; TCP/IP với robot chưa được triển khai |
| Truy vết | Không công bố hash/sealed manifest | Hash dữ liệu/model/config, sealed-open receipt, report manifest, 24 regression tests | Mới về reproducibility và audit |

## 3. Phần nào thực sự bám sát bài báo

Các thành phần sau có thể gọi là tái tạo hoặc kế thừa có cơ sở:

- Cùng bài toán inverse localization từ chín kênh cảm ứng điện từ sang sáu
  thành phần pose.
- Cùng mô hình vật lý point-dipole làm nền cho forward model.
- Cùng họ mạng fully connected residual, chiều rộng 512, bảy residual block và
  LeakyReLU.
- Cùng mục tiêu realtime batch-one và có đo latency trên GPU.
- Có calibration bằng dữ liệu thật và thử đưa tham số calibration trở lại
  pipeline huấn luyện.
- Có báo cáo position RMSE theo dạng Eq. (8) và Euler-component direction RMSE
  gần với Eq. (9), đồng thời bổ sung SO(3) geodesic.

Những điểm chỉ “lấy cảm hứng” chứ không phải tái tạo y nguyên:

- Dữ liệu synthetic khác số lượng và phân bố.
- Cửa sổ W3 thay cho single-row.
- Rotation-6D thay cho cosine Euler.
- Ensemble năm model thay cho một model.
- Grouped session CV và sealed final thay cho protocol của bài báo.
- Hình 4 và Hình 7 hiện tại là **structural analogue**, không phải lặp lại cùng
  ablation/calibration experiment.

### 3.1. Dữ liệu mô phỏng của bài báo được dùng để làm gì

Bài báo bắt đầu từ một forward model vật lý đã biết. Họ lấy lưới `40 × 40 ×
40` vị trí trong workspace 500 mm và `10 × 10 × 10` tổ hợp hướng, tổng cộng 64
triệu pose. Với từng pose, phương trình point-dipole sinh ra chín EMF. Sau đó
họ đảo cặp dữ liệu thành:

```text
input  = 9 EMF mô phỏng
target = X, Y, Z, cos(roll), cos(pitch), cos(yaw)
```

Vì vậy, synthetic data trong bài báo là **nguồn huấn luyện inverse model chính**
và có nhiệm vụ phủ dày toàn bộ không gian vị trí–hướng. Dữ liệu real 150 pose
được dùng để hiệu chỉnh forward model; các tham số đã hiệu chỉnh được đưa trở
lại quá trình retrain nhằm giảm sim-to-real gap.

### 3.2. Dữ liệu mô phỏng của ta thực sự được tạo như thế nào

Deployment được chọn `v3.2.2` **không dùng dữ liệu mô phỏng**. Nó học trực tiếp
từ 6.019 row robot-calibration thật thuộc chín session non-test. Synthetic chỉ
được đưa vào candidate v4 C3 theo quy trình sau:

1. Trong mỗi CV fold, fit một forward model chỉ từ các session train của fold.
2. Lấy pose hiện tại ngẫu nhiên đều trong cube 100 mm.
3. Lấy orientation ngẫu nhiên đều trong đúng min/max góc của train, cộng biên
   `±2°`; do dữ liệu train có orientation hẹp nên synthetic cũng hẹp.
4. Sinh một vận tốc tuyến tính ngẫu nhiên cho cửa sổ W3, tối đa `2 mm/row` và
   `1°/row`; hai pose quá khứ được suy ngược từ pose hiện tại rồi clip vào
   workspace.
5. Chạy cả ba pose qua forward model và channel correction để sinh `3 × 9`
   EMF; amplitude âm bị clamp về 0.
6. Gắn hai `delta-row = 1`, tạo packed W3 29 chiều. C3 tự tạo thêm 18 log-ratio
   bên trong model, thành 47 feature.
7. Sinh số synthetic bằng số real train (`1:1`), trộn ngẫu nhiên với real.
   Validation vẫn 100% real.
8. Sau pretrain real+synthetic, fine-tune lần hai trên real-only và tính lại
   normalization từ real.

Ở deployment v4 đã bị loại, Stage 1 dùng `6.019 real + 6.019 synthetic`; Stage
2 dùng lại `6.019 real`. Đây là random local linear W3 augmentation, không phải
lưới 64 triệu pose và cũng không tạo ra orientation coverage mà robot chưa đo.

### 3.3. Dữ liệu calibration của ta được dùng theo ba vai trò

Từ “calibration data” trong pipeline hiện tại dễ gây nhầm vì cùng file robot
được dùng cho ba mục đích hợp lệ nhưng khác nhau:

1. **Supervised inverse training:** cặp `(EMF thật, pose robot)` trực tiếp huấn
   luyện v3.2.2. Đây là vai trò chính của dữ liệu trong model đang triển khai.
2. **Physical system identification của v4:** một tập con train được dùng để
   fit forward model. Trong CV, held-out session không tham gia bước fit.
3. **Real-only domain anchoring:** sau khi pretrain hybrid, toàn bộ train real
   được dùng để fine-tune, kéo model trở lại miền sensor thật.

Final `cyl_rot` không được dùng cho ba vai trò này trước lúc mở final. Sau khi
đã mở, nó chỉ còn là historical reporting set, không được trở thành calibration
data cho model thay đổi tiếp theo.

### 3.4. Mục đích có giống bài báo không

| Câu hỏi | Trả lời |
|---|---|
| Có cùng ý tưởng dùng vật lý để tạo cặp pose–EMF không? | Có, trong candidate v4 |
| Synthetic có cùng vai trò huấn luyện chính không? | Không; bài báo dùng 64 triệu mẫu làm nền chính, ta chỉ dùng 1:1 như augmentation rồi fine-tune real |
| Real calibration có cùng mục tiêu giảm sim-to-real gap không? | Có ở mức mục tiêu |
| Cơ chế có giống không? | Chỉ một phần; ta vừa fit physical parameters, vừa học neural adapter, vừa supervised-train trực tiếp trên real |
| Đã đạt cùng kết quả physics-consistency chưa? | Không; final rotating reconstruction thất bại rõ rệt |

Nói chính xác nhất: ta giữ đúng triết lý “real calibration hiệu chỉnh mô hình
vật lý rồi hỗ trợ inverse network”, nhưng chuyển trọng tâm sang real-supervised
learning và dùng synthetic như regularizer. Mục đích tương tự, vai trò, quy mô
và mức độ thành công chưa giống bài báo.

### 3.5. Vì sao không tăng mạnh synthetic sau khi calibration

Ta đã thử đúng chuỗi `fit calibration → sinh synthetic → train → real-only
fine-tune` trong v4. Stage 1 tăng từ 6.019 lên 12.038 mẫu. Về kỹ thuật có thể
sinh hàng trăm nghìn hoặc hàng triệu mẫu, nhưng các mẫu đó không tạo thêm phép
đo vật lý độc lập. Chúng đều là đầu ra của cùng một forward model đã fit.

Nếu hệ thật là `s = F_true(p)` nhưng calibration chỉ học được
`s = F_fit(p)`, synthetic vô hạn chỉ làm inverse network học rất chính xác
`F_fit⁻¹`; nó không tự tiến gần `F_true⁻¹`. Khi `F_fit` có systematic bias,
tăng tỷ lệ synthetic còn có thể làm model xa miền real hơn.

Đó chính là rủi ro hiện tại:

- Ground-truth pose qua calibrated forward model vẫn có mean MAPE `106.19%`
  trên rotating final.
- Fitted channel gain có giá trị chạm bound `20.0`, bias tới `-45.34 mV`.
- Hardware channel mapping, polarity và pose frame chưa được xác minh.
- Orientation range của generator lấy từ train nên vẫn hẹp; mở rộng góc ngoài
  vùng calibration chỉ là extrapolation không có bằng chứng.

Kết quả thực nghiệm cũng phản ánh điều này. Direct mixed training v4.1 không
tốt; staged v4.2 cải thiện development và orientation final nhưng làm position
final tăng từ `1.1222` lên `1.2335 mm`, position p95 từ `3.1047` lên
`3.4725 mm`. Vì thế synthetic hiện chỉ được coi là regularizer thử nghiệm, chưa
đủ tin cậy để thay real data hoặc trở thành model triển khai.

“Chuẩn hóa bằng calibration” cũng cần tách thành ba thao tác:

1. **Statistical normalization:** mean/std chỉ tính trên train để đưa feature
   về thang số ổn định. Việc này đã làm ở v3/v4 nhưng không sửa physics bias.
2. **Physical channel correction:** forward EMF được nhân gain và cộng bias đã
   fit. Việc này chỉ đáng tin nếu mapping đúng và generalize theo pose.
3. **Neural adapter:** C3 học gain/bias theo generation, sau đó Stage 2 rebase
   normalization trên real-only. Việc này đã thử nhưng candidate vẫn fail
   balanced final gate.

Do đó hướng đúng không phải “không dùng synthetic”, mà là **không mặc định
synthetic càng nhiều càng tốt**. Sau kết quả final, ta vẫn có thể làm thí
nghiệm mới trên `con_rot` development đã có, miễn là không đọc lại `cyl_rot` để
chọn model. Thí nghiệm tỷ lệ ở mục 3.6 đã làm đúng theo ranh giới này. Tỷ lệ
`2×` chưa được chạy vì ngay cả `0.25×` và `0.50×` cũng không vượt đồng thời
`1×` và không qua session gate; tăng tiếp khi forward model chưa ổn định không
có đủ căn cứ.

### 3.6. Thử nghiệm kết hợp synthetic + calibration thật

Ta giữ nguyên pipeline C3 staged, seed `42` và rule blend đã chọn trước đây:

```text
Stage 1: real calibration train + physics-synthetic train
Stage 2: fine-tune chỉ bằng real calibration train, rebase normalization
Validation: chỉ real held-out con_rot; không có synthetic
Position: 0.8 * v3.2.2 + 0.2 * C3
Orientation: mean rotation-6D của v3.2.2 và C3
```

Chỉ thay tỷ lệ số synthetic row / real row. Mỗi tỷ lệ được train lại trên ba
leave-one-session-out fold. `cyl_rot` không được loader của thí nghiệm này đọc
hay chấm lại.

| Synthetic / real | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 | Composite gain so với v3.2.2 | Worst session p95 regression | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0, baseline v3.2.2 | 0.7474 mm | 2.3539 mm | 0.4602° | 0.8580° | 0.00% | 0.00% | reference |
| 0.25 | 0.7401 mm | 2.2653 mm | 0.3588° | **0.6242°** | **14.26%** | **+8.08%** | reject |
| 0.50 | **0.7298 mm** | 2.3057 mm | 0.3798° | 0.6594° | 11.75% | **+6.13%** | reject |
| 1.00 | 0.7327 mm | **2.2613 mm** | **0.3586°** | 0.6362° | 14.12% | +2.71% | pass development |

Hai tỷ lệ mới đều cải thiện cả bốn metric aggregate so với matched-fold
v3.2.2. Tuy nhiên, `0.25×` làm SO(3) p95 của session s3 tăng `8.08%`, còn
`0.50×` tăng `6.13%`; cả hai vượt giới hạn `5%`. Chúng cũng không thống trị
`1×`: `0.50×` có position RMSE tốt nhất, `0.25×` có orientation p95 tốt nhất,
nhưng `1×` cân bằng tốt hơn và là tỷ lệ duy nhất qua development gate.

Kết luận không phải synthetic vô ích. Nó có tín hiệu regularization rõ trên
aggregate orientation. Nhưng hiệu quả không đơn điệu theo số mẫu và còn phụ
thuộc session, nên chưa thể thay dữ liệu thật. Cấu hình `1×` từng qua
development rồi đã thất bại balanced final vì position; không được dùng hai
tỷ lệ mới để chọn lại rule dựa trên final đã mở. Deployment vẫn là v3.2.2.

![Ablation tỷ lệ synthetic](../reports/synthetic_ratio_ablation_2026_08_04/synthetic_ratio_ablation_seed42.png)

Artifact tái lập:

- [Báo cáo ablation](../reports/synthetic_ratio_ablation_2026_08_04/REPORT.md)
- [Bảng Markdown](../reports/synthetic_ratio_ablation_2026_08_04/synthetic_ratio_ablation_seed42.md)
- [Manifest JSON](../reports/synthetic_ratio_ablation_2026_08_04/synthetic_ratio_ablation_seed42.json)
- `evaluation/summarize_synthetic_ratio_ablation_v4.py`

### 3.7. Thử đúng thứ tự synthetic-only rồi calibration thật

Nhận xét “bài báo synthetic trước rồi mới calibration” là có cơ sở, nhưng cần
diễn đạt chính xác. Bài báo trước tiên xây tập analytical synthetic `64 triệu`
mẫu và huấn luyện inverse model. Ở hệ thật, họ thu `150` pose ngẫu nhiên, fit
lại vị trí, hướng và số vòng hiệu dụng của TX, rồi đưa tham số calibrated vào
forward/inverse models để retrain. Bài báo không công bố đủ chi tiết retrain
dùng bao nhiêu cặp real trực tiếp hay tái sinh bao nhiêu synthetic.

Ta đã bổ sung một ablation gần thứ tự neural training này hơn:

```text
Mỗi fold:
  fit/gate forward physics chỉ bằng train-fold real
  Stage 1: 107 epoch, chỉ physics-synthetic, không real, không dùng val chọn epoch
  Stage 2: chỉ real calibration, reset normalization, chọn checkpoint trên held-out real
  Evaluation: held-out con_rot session
  Final cyl_rot: không đọc, không chấm lại
```

Điểm không được đánh tráo: synthetic của ta đã dùng forward model calibrated
theo train fold trước Stage 1. Vì vậy đây chứng minh lợi ích của thứ tự
**synthetic-only pretrain → real-only neural calibration**, chưa phải tái tạo
đúng thứ tự nominal-physics → physical calibration → synthetic retraining của
bài báo.

Seed `42`, giữ rule equal-orientation cũ, cho:

| Protocol | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 |
|---|---:|---:|---:|---:|
| v3.2.2 matched baseline | 0.7474 mm | 2.3539 mm | 0.4602° | 0.8580° |
| Mixed real+synthetic `1×`, rồi real | 0.7327 mm | 2.2613 mm | **0.3586°** | **0.6362°** |
| Synthetic-only `1×`, rồi real | **0.6878 mm** | **2.0470 mm** | 0.3800° | 0.6892° |

Thứ tự mới cải thiện position rõ so với mixed training, nhưng equal
orientation `50/50` nhạy seed: seed `7` và `123` làm session-s3 orientation p95
tăng `9.06%` và `17.57%`. Vì vậy rule được giảm contribution orientation của
C3 xuống `30%`, hoàn toàn bằng development data:

```text
XYZ        = 0.8 * v3.2.2 + 0.2 * C3
rotation6D = 0.7 * v3.2.2 + 0.3 * C3, sau đó Gram–Schmidt
```

| Seed | Position RMSE | Position p95 | SO(3) RMSE | SO(3) p95 | Composite gain | Worst session p95 regression | Accuracy/session gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.6878 mm | 2.0470 mm | 0.4048° | 0.7384° | 11.78% | -0.69% | pass |
| 7 | 0.7077 mm | 2.1137 mm | 0.4045° | 0.7495° | 10.12% | +1.78% | pass |
| 123 | 0.7021 mm | 2.0971 mm | 0.4096° | 0.7395° | 10.49% | +1.54% | pass |

Đây là kết quả development đa seed tốt và ổn định hơn. Research deployment
checkpoint đã được train bằng `6.019` synthetic-only row trong 107 epoch, sau
đó `6.019` real-only row trong 108 epoch. CPU/GPU parity max difference là
`0.00003052`; đổi absolute timestamp không đổi pose.

Tuy nhiên candidate chưa qua deployment gate đầy đủ: ba lần benchmark cho
median CPU p95 `2.1648 ms` (qua 3 ms), GPU p95 `1.7234 ms` (không qua 1.5 ms).
Checkpoint v4 cũ benchmark lại cùng thời điểm cũng đạt GPU p95 `1.6388 ms`, cho
thấy host hiện tại có jitter so với số lịch sử `1.4226 ms`; ngưỡng khai báo vẫn
không được nới sau khi thấy kết quả. Quan trọng hơn, không còn sealed test mới.
Vì vậy đây là **research candidate**, deployment vẫn là v3.2.2.

![Synthetic rồi calibration](../reports/synth_then_calib_2026_08_05/fig_synth_then_calib_ablation.png)

Artifact:

- [Báo cáo thí nghiệm](../reports/synth_then_calib_2026_08_05/REPORT.md)
- [Manifest](../reports/synth_then_calib_2026_08_05/manifest.json)
- [Research checkpoint manifest](../checkpoints/new_calib_v4_synth_then_calib/C3/research_candidate_manifest.json)
- `evaluation/summarize_synth_then_calib_v4.py`

### 3.8. Audit chất lượng dữ liệu calibration hiện tại

Kiểm tra 12 file theo cùng programmed-row cho thấy robot label có độ lặp lại
rất tốt: median position difference giữa session chỉ `0.01–0.014 mm`, p95 lớn
nhất khoảng `0.193 mm`; Euler-component difference p95 chỉ `0.008–0.012°`.
Điều này ủng hộ việc dùng pose robot làm supervised target.

Ngược lại, chín kênh EMF tại cùng programmed row/pose thay đổi đáng kể giữa
session:

| Family | So sánh | Position p95 | EMF normalized RMS difference | EMF symmetric MAPE |
|---|---|---:|---:|---:|
| `con_norot` | s1–s2 / s1–s3 | 0.095 / 0.126 mm | 0.226 / 0.221 | 21.28 / 21.33% |
| `con_rot` | s1–s2 / s1–s3 | 0.024 / 0.022 mm | 0.218 / 0.087 | 25.32 / 14.28% |
| `cyl_norot` | s1–s2 / s1–s3 | 0.189 / 0.193 mm | 0.152 / 0.262 | 18.91 / 23.78% |
| `cyl_rot` final | s1–s2 / s1–s3 | 0.024 / 0.024 mm | 0.150 / 0.259 | 18.63 / 23.31% |

Sai khác EMF `14–25%` có thể đến từ gain/offset drift, nhiệt độ, lắp capsule,
môi trường kim loại hoặc độ trễ đồng bộ EMF–pose. Same-row audit chưa tách được
các nguyên nhân đó, nên không nên diễn giải toàn bộ là sensor noise ngẫu nhiên.

Về coverage, 6.019 row train chiếm 102/125 cell của lưới không gian `5³`, nhưng
số hàng mỗi cell rất lệch: min 9, median 21, max 897. Chỉ có khoảng 2.345 vị trí
khác nhau khi làm tròn 0.1 mm vì các session lặp quỹ đạo. Orientation train chỉ
phủ roll `0.02°`, pitch `10°`, yaw `0.04°`.

Đánh giá phù hợp nhất:

| Khía cạnh | Mức hiện tại |
|---|---|
| Độ chính xác/lặp lại pose robot | Tốt |
| Phủ vị trí trong cube | Khá nhưng mất cân bằng và lặp đường |
| Phủ orientation 6-DoF | Kém; chủ yếu pitch |
| Độ lặp lại EMF xuyên session | Chưa tốt |
| Học supervised trong cùng setup | Có ích; final position RMSE 1.1222 mm |
| System identification/physics generator | Chưa đủ tin cậy |
| Khẳng định generalization xuyên ngày/phần cứng | Chưa có bằng chứng |

Do đó dùng calibration data là lựa chọn đúng và hiện tốt hơn synthetic đối với
position, nhưng bộ hiện tại nên được gọi là **usable in-domain supervised
calibration data**, chưa phải **broad, physically stable 6-DoF calibration
dataset**.

## 4. Những đóng góp mới của pipeline hiện tại

### 4.1. Temporal feature nhân quả

Mỗi dự đoán dùng row hiện tại và hai row quá khứ. Cửa sổ không vượt session,
đoạn quỹ đạo hoặc reset. Model không thấy future row. Với dữ liệu hiện tại,
hai giá trị thời gian chỉ là `delta-row = 1`; padding sau reset là `0`.
Timestamp tuyệt đối và row index tuyệt đối không được dùng làm feature.

Điểm mới này có ích cho realtime vì EMF liên tiếp mang thông tin chuyển động,
nhưng chưa chứng minh được lợi ích của thời gian vật lý do acquisition hiện
không có timestamp phần cứng và tốc độ row được lập trình cố định.

### 4.2. Feature log-ratio và ensemble tách nhiệm vụ

Deployment v3.2.2 dùng:

```text
XYZ = mean(raw-W3 seed42, raw-W3 seed7, raw-W3 seed123,
           unit-row-time W3 seed42)

orientation = current EMF + log(past/current) W3 seed42
```

Đây là thay đổi quan trọng so với một network duy nhất của bài báo. Các nhánh
được chọn trên development CV, không chọn trên final `cyl_rot`.

### 4.3. Rotation-6D và metric SO(3)

Khác biệt neural lớn nhất ở head orientation. Bài báo xuất ba giá trị
`cos(roll), cos(pitch), cos(yaw)`. Trong miền góc bị giới hạn `[0, 180°]` của
bài báo, `arccos` có thể khôi phục góc, nên lựa chọn đó hợp lý cho protocol của
họ. Ngoài miền này, cosine có hai vấn đề: `cos(θ) = cos(-θ)` làm mất dấu và ba
Euler coordinate không biểu diễn trực tiếp khoảng cách hình học giữa hai phép
quay.

Pipeline hiện tại trước tiên tạo ma trận:

```text
R = Rz(yaw) · Ry(pitch) · Rx(roll)
```

Target neural là hai cột đầu của `R`, tức sáu số. Model dự đoán hai vector tự
do `a1, a2`; Gram–Schmidt biến chúng thành:

```text
b1 = normalize(a1)
b2 = normalize(a2 - dot(b1, a2) b1)
b3 = cross(b1, b2)
R_pred = [b1 b2 b3]
```

Nhờ đó đầu ra luôn được chiếu về một ma trận quay proper trên SO(3). Loss hướng
là bình phương geodesic angle theo `R_predᵀR_true`, tính bằng `atan2` để gradient
ổn định gần 0. Evaluation dùng:

```text
angle = acos((trace(R_predᵀ R_true) - 1) / 2)
```

Đổi mới này có bốn lợi ích:

- Giữ được hướng quay có dấu.
- Không bị nhảy loss trực tiếp tại biên Euler `-180°/180°`.
- Một sai số quay được đo độc lập với cách tham số hóa Euler.
- Có thể average nhiều model trong không gian rotation rồi project về SO(3),
  thay vì lấy trung bình từng góc một cách không hợp lệ.

Giới hạn: rotation-6D không bù được orientation coverage nghèo. Với roll/yaw
gần như cố định trong dữ liệu hiện tại, model vẫn chưa được chứng minh trên
full dynamic 6-DoF. Khi xuất roll/pitch/yaw cho robot, singularity của Euler
vẫn có thể xuất hiện ở bước chuyển đổi cuối.

### 4.4. Mạng neural thực sự khác bài báo ở đâu

| Thành phần | Bài báo | v3.2.2 được chọn | Candidate v4 C3 |
|---|---|---|---|
| Số model | 1 | 5 model | 1 C3 cộng với v3.2.2 khi blend |
| Input model | 9 EMF của một pose | raw-W3 27; rowtime 29; log-ratio branch 27 | packed W3 29, tạo nội bộ 47 feature |
| Temporal | Không | Hai row quá khứ, nhân quả | Hai row quá khứ, nhân quả |
| Residual trunk | Width 512, 7 block, mỗi block 2 linear, LeakyReLU 0.01 | Giữ cùng lõi trunk | Giữ cùng lõi trunk |
| Output | 6: XYZ + 3 cosine Euler | 9: XYZ + rotation-6D | 9: XYZ + rotation-6D |
| Loss | MSE trên vector pose | normalized position MSE + squared SO(3) geodesic | Cùng loss v3 + adapter regularization |
| Data | Chủ yếu synthetic 64 triệu | 6.019 real row, không synthetic | Real + synthetic 1:1, sau đó real-only fine-tune |
| Robustness | Không báo cáo gain/noise augmentation | Batch-coherent gain drift ±1%, noise 0.2 mV | ±1%/0.2 mV rồi ±0.5%/0.1 mV khi fine-tune |
| Selection | Validation/test theo protocol bài báo | Branch/seed/epoch từ grouped CV | Bốn-metric composite và final gate |

Do đó câu trả lời là: ta không thay hoàn toàn ResNet của bài báo; ta giữ phần
thân residual đã hợp lý, nhưng thay đổi representation, input temporal, loss,
quy tắc ensemble, data protocol và deployment runtime. Phần đổi mới nằm nhiều
ở “hệ thống quanh network” và orientation geometry hơn là phát minh một loại
block neural mới.

### 4.5. Chống leakage theo session và cửa sổ

- Chia session trước khi tạo cửa sổ temporal.
- Normalizer, median delta-time và physics calibration chỉ fit trên fold train.
- Synthetic chỉ thêm vào train; validation luôn là real.
- `cyl_rot` chỉ được materialize sau khi candidate/config/hash đã freeze.
- Final test đã mở đúng một lần; từ thời điểm đó không được dùng để chọn blend
  hoặc component mới.

Đây là cải tiến phương pháp luận so với random row split trong dữ liệu quỹ đạo.
Tuy nhiên, không được tuyên bố “không còn bất kỳ leakage nào”; phần 9 nêu các
rủi ro còn lại.

### 4.6. Calibration vật lý và neural adapter

Calibration v4 có hai tầng, cần tách rõ:

**Tầng physical fit.** Mỗi fold lấy mẫu cân bằng giữa các source, tối đa 1.200
row; deployment lấy 1.800 row sau khi ba fold đều qua gate. SciPy
`least_squares` dùng Trust Region Reflective, `soft_l1`, bốn điểm khởi tạo và
residual được chia theo độ lệch chuẩn từng kênh. Tổng cộng 39 tham số được fit:

- 3 vị trí × 3 TX = 9 tham số.
- 3 vector hướng × 3 TX = 9 tham số, normalize về unit vector.
- 3 magnetic moment dương, tham số hóa log.
- 9 channel gain dương, tham số hóa log.
- 9 channel bias.

Tần số TX `4/4.5/5 kHz`, RX area và RX turns được giữ cố định. Bài báo fit vị
trí, hướng và effective turns của TX; moment của ta đóng vai trò scale vật lý
tương tự, nhưng chín gain/bias là phần empirical bổ sung nên physical
interpretability thấp hơn.

Generator chỉ được dùng nếu held-out correlation ≥0.80 và
`RMSE/mean(|EMF|) ≤ 0.50`. Các fold đạt correlation `0.878–0.918` và tỷ lệ lỗi
`0.390–0.474`. Gate này chỉ đủ để cho phép một thí nghiệm augmentation; nó
không gần mức physics-consistency `<3.4%` của bài báo.

**Tầng neural adapter.** C3 có một gain và bias học được cho mỗi trong chín
kênh, riêng cho từng generation `real_current` và `physics_synthetic`. Cùng một
gain/bias được áp dụng cho cả ba row W3 để không tạo temporal artifact. Gain
được tham số hóa bằng `exp(log_gain)` nên luôn dương; bias được scale theo độ
lệch chuẩn kênh; tất cả khởi tạo identity và có L2 regularization `0.001`.

Trong checkpoint v4 deployment đã bị loại, adapter miền real học gain khoảng
`0.833–1.051` và bias khoảng `-0.823 đến +4.515 mV`. Các số này là tham số
end-to-end, không nên diễn giải như phép đo phần cứng độc lập vì trunk neural có
thể bù trừ ngược lại.

Forward calibration deployment còn có dấu hiệu thiếu định danh: channel gain
fit trải từ khoảng `0.636` đến `20.0`, trong đó một kênh chạm upper bound và một
kênh gần upper bound; bias trải từ khoảng `-45.34` đến `+13.90 mV`. Điều này cho
thấy optimizer đang dùng empirical correction khá mạnh để bù mismatch. Cùng
với ba cờ hardware mapping/polarity/frame chưa xác minh, đây là lời giải thích
hợp lý cho thất bại Fig. 9 trên rotating final.

### 4.7. Physics-hybrid theo hai giai đoạn

Candidate v4 C3 thực hiện:

1. Fit forward model riêng trong mỗi fold.
2. Chỉ cho phép sinh synthetic nếu correlation và normalized RMSE qua gate.
3. Pretrain với real + synthetic tỷ lệ 1:1.
4. Fine-tune bằng real-only và rebase normalization.
5. Dùng adapter gain/bias theo generation và spatial/session sampler.

Cách này cải thiện development CV nhưng không thắng final. Do đó đây là một
thử nghiệm mới có kết quả âm/mixed, không phải mô hình triển khai.

### 4.8. Quality gate nhiều mục tiêu

Candidate chỉ được chấp nhận nếu đồng thời cải thiện position RMSE, position
p95, SO(3) RMSE, SO(3) p95; composite cải thiện ít nhất 3%; không session nào
regress p95 quá 5%; latency CPU/GPU nằm trong ngưỡng. Gate này đã hoạt động
đúng mục đích: v4 có orientation tốt hơn nhưng position xấu hơn, nên bị loại.

### 4.9. Runtime và interface realtime

v3.2.2 hỗ trợ TorchScript CPU fallback và CUDA Graph GPU. JSONL predictor có
state W3, explicit reset, tự reset khi timestamp không tăng, và hỗ trợ ngưỡng
gap tùy chọn nếu caller cấu hình. Lớp TCP tương lai phải gửi reset sau khi mất
kết nối. Confidence/dispersion được trả cùng pose nhưng hiện là heuristic dựa
trên độ phân tán ensemble/OOD feature, chưa phải uncertainty đã calibration.
TCP/IP transport với robot mới ở mức interface design, chưa phải phần đã hoàn
thiện hoặc benchmark.

## 5. Đối chiếu từng hình của bài báo

| Hình bài báo | Nội dung bài báo | Trạng thái trong workspace |
|---|---|---|
| Fig. 1 | Toàn hệ thống capsule, TX, RX, wireless và bộ xử lý | Không tái tạo thành sơ đồ/hardware report hoàn chỉnh |
| Fig. 2 | Hệ tọa độ TX/RX và hình học forward model | Có code forward model, nhưng ba cờ `channel_mapping`, `channel_polarity`, `pose_frame_transform` vẫn chưa được xác minh phần cứng |
| Fig. 3 | Một residual FCN width 512, bảy block | Tái tạo lõi kiến trúc; deployment mở rộng thành ensemble temporal năm model |
| Fig. 4 | FCN vs residual, raw angle vs cosine, Tanh vs LeakyReLU, số block | Chưa lặp lại đúng bốn ablation; hình hiện tại là analogue về staged training và seed robustness |
| Fig. 5 | Simulation: helix cố định, helix đổi hướng, random pose; so với optimization | Chưa tái tạo trên cùng protocol và chưa có optimization baseline hiện tại |
| Fig. 6 | Ảnh experimental setup thật | Không có deliverable tương đương trong workspace |
| Fig. 7 | Sai số trước/sau physical calibration | Có analogue trên fold s3, nhưng là real-only C3 so với physics-pretrain + real-finetune |
| Fig. 8 | Real benchmark 100 pose trên tapered helix, orientation rộng | Có analogue 100 điểm từ quỹ đạo cylinder hiện tại; hình học và orientation khác rõ rệt |
| Fig. 9 | EMF reconstruction gần trùng measured, tất cả kênh <3.4% | Đã kiểm tra và **không đạt**; ground-truth control cho thấy lỗi thuộc forward calibration |

## 6. Mô tả từng hình hiện tại

### Hình 4 analogue — staged training và seed robustness

![Hình 4 analogue](../reports/paper_style_2026_08_04/figures/fig4_ablation_and_seed_robustness.png)

- Panel (a) vẽ position RMSE theo epoch của giai đoạn real + physics synthetic
  và giai đoạn fine-tune real-only.
- Panel (b) vẽ SO(3) RMSE của hai giai đoạn tương tự.
- Panel (c) đặt train loss và validation loss của real-only fine-tune trên thang
  log để quan sát generalization.
- Panel (d) so sánh candidate frozen `w=0.20` ở seed 42, 7, 123 với đường
  baseline v3.2.2.

Điều hình này cho phép nói: candidate v4 ổn định trên ba seed trong development
CV. Hình **không** cho phép nói ta đã lặp lại ablation FCN/residual,
cosine/raw-angle, activation hoặc residual depth của bài báo. Hai đường
pretrain/fine-tune cũng là hai stage có epoch counter riêng, không phải một
training curve liên tục.

### Hình 7 analogue — tác động của physics-guided calibration trên fold s3

![Hình 7 analogue](../reports/paper_style_2026_08_04/figures/fig7_physics_calibration_effect.png)

Panel trái là Euclidean position error theo row; panel phải là SO(3) error.
Đường mờ là lỗi từng row, đường đậm là rolling median chỉ để nhìn xu hướng.

| Fold s3 | Real-only C3 | Physics pretrain + real fine-tune | Thay đổi |
|---|---:|---:|---:|
| Position axis RMSE | 0.8083 mm | 0.6706 mm | tốt hơn 17.0% |
| Position Euclidean p95 | 2.1542 mm | 1.8644 mm | tốt hơn 13.4% |
| SO(3) RMSE | 0.2440° | 0.2386° | tốt hơn 2.2% |
| SO(3) p95 | 0.3234° | 0.3874° | **xấu hơn 19.8%** |

Vì orientation p95 xấu hơn, không được mô tả Fig. 7 hiện tại là “calibration
cải thiện mọi metric”. Nó cũng không tương đương phép so sánh trước/sau
hardware calibration của bài báo (`8.42 → 1.90 mm`, `12.02° → 3.55°`).

### Hình 8 analogue — final trajectory và lỗi không smoothing

![Hình 8 analogue](../reports/paper_style_2026_08_04/figures/fig8_final_trajectory_orientation_errors.png)

- Panel (a) hiển thị 100 row lấy đều từ session `cyl_rot/3`, gồm ground truth
  và dự đoán v3.2.2. Tọa độ chỉ được tịnh tiến về hệ local và đổi mm sang m;
  hình học không bị biến dạng.
- Panel (b) so sánh roll/pitch/yaw target và v3.2.2. Roll và yaw gần như phẳng
  vì đó là dữ liệu robot thật, không phải lỗi plotting.
- Panel (c) vẽ position error thô của v3.2.2, v4 component và frozen blend
  `w=0.20`; không dùng rolling smoothing.
- Panel (d) vẽ SO(3) error thô của ba phương án tương tự.

Bài báo dùng 100 pose trên tapered five-loop helix: phía dưới lớn, phía trên
nhỏ. Dữ liệu ta là five-loop cylinder với bán kính `50.000 ± 0.038 mm`, nên
trên và dưới phải bằng nhau. Tạo hình thuôn từ dữ liệu hiện tại sẽ là trình bày
sai. Hình chỉ hiển thị 100 điểm cho dễ đối chiếu; final metrics bên dưới được
tính trên toàn bộ 3.012 row của ba session, không phải 100 điểm chọn để vẽ.

### Hình 9 analogue — physics-consistency audit

![Hình 9 analogue](../reports/paper_style_2026_08_04/figures/fig9_physics_consistency.png)

- Panel (a) so sánh EMF1 đo thật với EMF tái dựng từ ground-truth pose, pose
  v3.2.2 và pose v4.
- Panel (b) báo MAPE từng kênh cho cả ba nguồn pose.

| Pose đưa vào forward model | Mean MAPE chín kênh |
|---|---:|
| Ground truth | 106.19% |
| v3.2.2 | 107.16% |
| v4 component | 105.39% |

Ground-truth pose cũng tái dựng kém gần như pose dự đoán. Vì vậy lỗi chính
không thể quy cho inverse network; nó nằm ở forward parameter calibration,
channel correction/mapping, polarity hoặc pose-frame convention khi chuyển
sang rotating sessions. Hình này là bằng chứng **chống lại** tuyên bố đã đạt
physics-consistency như bài báo.

## 7. Mô tả các bảng hiện tại

### Table I analogue

[Mở bảng theo session](../reports/paper_style_2026_08_04/tables/table1_scenario_performance.md).
Bảng này báo từng fold `con_rot` và final `cyl_rot`, gồm position/SO(3)
RMSE, p95 và runtime. Nó không tương đương Table I của bài báo vì chưa có ba
scenario simulation giống hệt và chưa chạy nonlinear optimization.

### Table II analogue

[Mở bảng benchmark](../reports/paper_style_2026_08_04/tables/table2_real_system_benchmark.md).
Các dòng Optimization/FCN/KAN/paper residual được chép đúng từ Table II của
bài báo để tạo bối cảnh; chúng không phải kết quả chạy lại trên current data.
Cột `Directly comparable` vì vậy luôn là `No`.

### Table III analogue

[Mở bảng feature/method](../reports/paper_style_2026_08_04/tables/table3_method_comparison.md).
Bảng này phân biệt residual model bài báo, deployment v3.2.2 và candidate v4.
Mục “experimental v4” không được hiểu là model đang triển khai.

### Table IV bổ sung

[Mở bảng reconstruction từng kênh](../reports/paper_style_2026_08_04/tables/table4_physics_reconstruction_error.md).
Đây không phải bảng của bài báo; nó được thêm để công khai thất bại của
physics-consistency và tránh chọn một kênh đẹp để trình bày.

## 8. So sánh định lượng không đánh tráo protocol

| Hệ thống | Workspace | Position RMSE | RMSE/cạnh workspace | Direction Eq. (9) | SO(3) RMSE | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Bài báo residual | 500 mm | 1.90 mm | 0.38% | 3.55° | Không báo cáo | 0.82 ms reported |
| v3.2.2 final, được chọn | 100 mm | 1.1222 mm | 1.1222% | 1.3801° | 2.3904° | GPU p95 0.858 ms; CPU p95 1.290 ms |
| Frozen hybrid v4 final | 100 mm | 1.2335 mm | 1.2335% | 1.0769° | 1.8653° | GPU p95 1.423 ms; CPU p95 1.992 ms |

`1.1222 mm < 1.90 mm` không có nghĩa ta tốt hơn bài báo. Khi chỉ chuẩn hóa sơ
bộ theo cạnh workspace, bài báo là `0.38%` còn ta là `1.1222%`. Ngay cả tỷ lệ
này cũng chưa tạo phép so sánh công bằng vì orientation, trajectory, sensor,
noise, split và metric aggregation khác nhau. Direction của ta thấp hơn về số
tuyệt đối nhưng được đo trên orientation range hẹp hơn rất nhiều.

Latency v3.2.2 ở bảng trên là inference core/network ensemble, không gồm thu
nhận EMF hay TCP/IP. Benchmark v4 bao phủ nhiều phần predictor hơn. Vì phạm vi
đo không hoàn toàn giống nhau, hai latency nội bộ này cũng không nên dùng như
một so sánh tuyệt đối giữa kiến trúc.

## 9. Leakage audit và các giới hạn còn lại

### Những điểm đã kiểm soát

- Không có row/window chồng giữa session train và held-out session.
- Cửa sổ W3 reset tại biên file/đoạn 1.000 row.
- Không dùng XYZ, orientation, velocity hoặc distance-to-boundary làm feature.
- Không dùng timestamp tuyệt đối hoặc absolute row index.
- Synthetic không đi vào validation/final.
- Candidate và rule đã hash-freeze trước lần mở final.
- Script báo cáo hiện tại chỉ đọc artifact frozen, không train hoặc chọn lại.

### Những điểm không được che giấu

- Raw target của final tồn tại trong workspace, và một audit preprocessing cũ
  từng nhìn target-derived restart boundary. Vì vậy đây không phải sealed test
  độc lập ở mức máy chủ/đơn vị thứ ba, dù pipeline v4 có freeze manifest và
  one-time receipt.
- `con_rot` và `cyl_rot` dùng quỹ đạo robot gần giống nhau. Không có direct row
  leakage, nhưng model vẫn có thể khai thác cấu trúc quỹ đạo lặp lại.
- Orientation schedule có tương quan mạnh với kịch bản quét. Bỏ absolute time
  làm giảm shortcut nhưng không loại bỏ hoàn toàn shortcut từ chuỗi EMF.
- Final test đã mở; hiện không còn một test mới chưa nhìn thấy để xác nhận bất
  kỳ model thay đổi nào tiếp theo.
- Chọn v4 orientation component riêng sau khi thấy final sẽ là test-derived
  selection. Ý tưởng đó chỉ được thử lại trên development data mới.
- Ba cờ xác minh phần cứng trong `configs/new_calib_v3.yaml` vẫn là `false`:
  channel mapping, channel polarity và pose-frame transform.

Kết luận leakage phù hợp nhất là: **không phát hiện direct row/window leakage
trong split đã triển khai, nhưng vẫn còn nguy cơ distribution/trajectory
shortcut và độ kín final chưa đạt chuẩn đánh giá độc lập tuyệt đối**.

## 10. Điều có thể và không thể tuyên bố

### Có thể tuyên bố

- Đã triển khai một pipeline 6-DoF EMF residual có khả năng chạy realtime trên
  CPU hoặc GPU trong workspace 100 mm.
- Đã bổ sung temporal causal features, rotation-6D, grouped CV, sealed final,
  multi-seed ensemble và reproducibility manifest so với bài báo.
- v3.2.2 đạt các số final đã nêu trên protocol hiện tại.
- Physics-hybrid v4 cải thiện orientation nhưng thất bại balanced position gate.
- Physics forward model hiện không tái dựng tốt rotating final data.

### Không thể tuyên bố

- “Đã tái tạo toàn bộ bài báo.”
- “Đã vượt bài báo” về độ chính xác hoặc tốc độ.
- “Đã chứng minh 6-DoF dynamic rộng” tương đương `20°–160°` trên cả ba góc.
- “Đã đạt physics-consistency.”
- “Không có leakage tuyệt đối.”
- “TCP/IP robot integration đã hoàn thành.”
- “v4 là model tốt nhất” hoặc “nên dùng v4 cho deployment”.

## 11. Thí nghiệm cần làm tiếp theo để so sánh công bằng hơn

1. Thu ít nhất ba rotating development session mới, thay đổi rộng cả roll,
   pitch, yaw; thu thêm một session muộn hơn để sealed final.
2. Thêm tapered five-loop helix giống bài báo, random pose order, đảo chiều,
   thay đổi phase khởi đầu và tốc độ để giảm trajectory shortcut.
3. Ghi hardware timestamp và reset flag thật; chỉ dùng delta-time khi sampling
   interval thực sự biến đổi.
4. Xác minh vật lý channel mapping, polarity và pose-frame convention trước khi
   fit lại forward model.
5. Trên cùng development protocol mới, chạy single-row paper-like ResNet,
   plain FCN, KAN và nonlinear optimization. Khi đó mới có Table II ngang hàng.
6. Lặp đúng bốn ablation của Fig. 4: skip connection, orientation encoding,
   activation và residual depth.
7. Chỉ sau khi khóa toàn bộ rule mới mở sealed test mới đúng một lần.
8. Benchmark end-to-end gồm acquisition, feature construction, inference,
   JSON serialization và TCP/IP round trip; latency hiện tại chưa bao gồm toàn
   bộ chuỗi này.

## 12. Artifact và cách tái tạo báo cáo hình/bảng

```bash
# Chạy từ thư mục gốc của repository:
./.venv/bin/python reporting/build_paper_style_report.py \
  --out_dir reports/paper_style_2026_08_04
```

Artifact chính:

- [Paper-style report](../reports/paper_style_2026_08_04/REPORT.md)
- [Report manifest](../reports/paper_style_2026_08_04/report_manifest.json)
- [V4 runbook](V4_UPGRADE_RUNBOOK.md)
- [Experiment ledger](EXPERIMENT_LEDGER.md)
- [Model upgrade history](MODEL_UPGRADE_HISTORY.csv)
- [Final decision manifest](../results/new_calib_v4_final/final_decision_manifest.json)
- [Selected deployment manifest](../checkpoints/new_calib_v3_temporal_deployment/deployment_manifest.json)

Tất cả 26 regression tests hiện đều pass. Các hash trong report manifest xác
nhận đúng các file nguồn và file sinh báo cáo tại thời điểm lập tài liệu này.
