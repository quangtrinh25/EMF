| Hình paper | Nội dung paper | Đối ứng của ta | Mức tương đương trung thực |
|---|---|---|---|
| Fig. 1 | System overview | Fig01 pipeline side-by-side | Khái niệm tương ứng; chưa vẽ lại hardware geometry |
| Fig. 2 | TX/RX coordinates và configuration | Chưa có exact analogue | Cần đo/xác minh channel mapping, polarity, pose-frame trên rig |
| Fig. 3 | Residual network | Fig02 architecture side-by-side | Backbone rất gần; input/output/composition khác |
| Fig. 4 | Ablation FCN/residual, representation, activation, depth | Fig10 analogue | Ta có staged training + seed robustness; không tái chạy đúng ablation paper |
| Fig. 5 | Ba simulation trajectories | Fig04 data/coverage | Ta chưa tái tạo đúng ba scenario 500 mm |
| Fig. 6 | Real experimental setup | Chưa có photo/geometry analogue | CSV robot không thay cho bản vẽ setup đo đạc |
| Fig. 7 | Before/after physical calibration | Fig11 analogue | Ta so neural stage trên fold s3; không phải same calibration plot |
| Fig. 8 | Tapered five-loop trajectory + errors | Fig12 analogue | Ta vẽ 100 điểm thật, không smooth; path là cylinder constant radius nên hình khác |
| Fig. 9 | EMF reconstructed consistency <3.4% | Fig13 + Fig09 audit | Ta không đạt: GT-pose control ~106.19% mean-channel MAPE |
