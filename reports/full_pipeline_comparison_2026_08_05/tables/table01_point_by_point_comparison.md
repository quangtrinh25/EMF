| Hạng mục | Bài báo | Ta | Phân loại | Kết luận kiểm toán |
|---|---|---|---|---|
| 1. Bài toán | 9 biên độ EMF → vị trí và hướng 6-DoF | Cùng bài toán inverse localization | Gần giống | Lõi mục tiêu được tái tạo |
| 2. Số kênh | 3 TX × 3 RX = 9 EMF | 9 EMF | Gần giống | Cùng kích thước quan sát hiện tại |
| 3. Workspace | 500×500×500 mm | 100×100×100 mm | Khác do phần cứng | Cạnh nhỏ hơn 5×; thể tích nhỏ hơn 125× |
| 4. Forward physics | Mô hình cảm ứng điện từ giải tích | Cùng họ mô hình dipole/induction + hiệu chỉnh kênh | Gần về nguyên lý | Thông số và mapping phần cứng chưa tương đương |
| 5. Synthetic | 40³ vị trí × 10³ orientation = 64M | ~5k/fold hoặc 6,019 deployment; causal W3 | Khác lớn | Ta chưa tái tạo scale và coverage của paper |
| 6. Pose synthetic | Grid có cấu trúc phủ toàn workspace/orientation | Current pose uniform; history bằng bounded constant velocity | Khác | Phù hợp W3 realtime nhưng orientation bị giới hạn bởi calib |
| 7. Physical calibration | 150 pose thật fit vị trí/hướng/số vòng TX | Tối đa 1,200 row/fold; 1,800 deployment; fit 39 tham số | Cùng mục đích, khác cách | Đều giảm sim-to-real qua tham số vật lý |
| 8. Neural calibration | Calibrated parameters đưa vào forward/inverse rồi retrain; chi tiết dataset mơ hồ | Stage 1 synthetic-only → Stage 2 real-only, reset normalization | Analogue có kiểm soát | Cùng thứ tự cấp cao, không phải exact reproduction |
| 9. Calibration adapter | Không báo cáo | Gain dương + bias cho 9 kênh theo generation; identity init + regularization | Mới | Nhằm hấp thụ gain/bias giữa domain |
| 10. Input thời gian | Một row 9 EMF | Causal W3; v4 packed 27 EMF + 2 dt_ratio | Mới | Không dùng future row |
| 11. Timestamp | Không dùng | File hiện tại không có timestamp; mỗi row = 1 unit Δt | Khác | Không có timing biến thiên; absolute timestamp bị cấm |
| 12. Feature | Raw EMF | Raw + past/current log-ratio + relative Δt | Mới | Log-ratio giúp bớt nhạy scale; cần gate |
| 13. Backbone | Width 512, 7 residual blocks, 2 linear/block, LeakyReLU 0.01 | Giữ nguyên width/depth/block/slope | Rất gần | Đây là phần neural bám sát nhất |
| 14. Head | 512→128→6 | v3/v4: 512→128→9 | Khác | 9 = XYZ + rotation-6D |
| 15. Orientation target | cos(roll,pitch,yaw) | Rotation-6D → Gram–Schmidt → SO(3) | Mới | Tránh wrap và đánh giá hình học đúng hơn |
| 16. Loss | MSE | MSE trên target chuẩn hóa + adapter regularization | Gần + mở rộng | Không dùng p95/GT-derived feature lúc inference |
| 17. Một model/ensemble | Một ResNet đề xuất | Selected v3.2.2 có 5 ResNet; v4.4 blend frozen baseline + C3 | Khác | Độ chính xác đổi lấy latency/complexity |
| 18. Sampling | Sampling từ grid synthetic | Inverse-sqrt 5³ cell, cân bằng session/generation, cap 3× | Mới | Giảm thiên lệch vùng dày |
| 19. Split | Random 70/15/15 trên synthetic | Leave-one-session-out development; final family khóa | Chặt hơn về leakage | Đánh giá cross-session khó hơn random-row |
| 20. Cửa sổ | Không có | Tạo sau split; không vượt reset/session/generation | Chặt hơn | Ngăn adjacent-window leakage |
| 21. Normalization | Gaussian normalization | Train-fold feature/target normalization; reset khi real fine-tune | Gần + chặt hơn | Thống kê không lấy từ val/final |
| 22. Seed | Không thấy robustness nhiều seed trong kết quả chính | 42 screening; 7 và 123 confirmation | Chặt hơn | v4.4 qua accuracy/session gate ở cả ba |
| 23. Metric | Axis position RMSE; Euler-component direction RMSE | Thêm Euclidean p95, SO(3) RMSE/p95 theo session | Mở rộng | Tail error và rotation geometry rõ hơn |
| 24. Model gate | Chọn theo thí nghiệm/ablation được mô tả | Bốn metric cùng cải thiện + composite ≥3% + per-session ≤5% | Chặt hơn | Không chọn vì một metric đẹp |
| 25. Final test | 100 pose tapered five-loop helix | 3,012 row, 3 session cylinder constant radius | Khác | Fig. 8 của ta không thể có taper giống paper |
| 26. Final lock | Không công bố receipt/hash workflow | Hash freeze; cyl_rot mở một lần; cấm tune lại | Chặt hơn về protocol | v4.2 bị reject; v3.2.2 giữ deployment |
| 27. Physics consistency | EMF reconstruction <3.4% | GT-pose reconstruction mean channel MAPE 106.19% trên final quay | Chưa tái tạo | Forward calibration trên rotation là thiếu sót chính |
| 28. Runtime | 0.82 ms, RTX 3090 + i9-10980XE | v3.2.2: GPU p95 0.858 ms; CPU p95 1.290 ms | Gần về latency, không direct | Khác hardware/statistic/scope |
| 29. Realtime | PC localization real time | Stateful predictor `{timestamp_ns, reset, emf[9]}`; CPU/GPU fallback | Mở rộng triển khai | Acquisition/TCP/IP chưa nằm trong benchmark |
| 30. Robot/TCP-IP | Không phải đóng góp chính | Interface dự kiến model ↔ robot, chưa triển khai loop điều khiển | Chưa hoàn tất | Giữ ngoài training để tránh trộn scope |
