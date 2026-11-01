# Kết quả SAM–NPP đã hoàn tất: 02/11/2026

Cả bốn worker đã COMPLETE, exit code 0. Hai nhánh SAM trên mỗi worker đều huấn luyện đủ 768 bước và đánh giá đủ bốn QP. Không nhánh NPP/SAM nào đạt −15% BD-rate; Gaussian tốt nhất vẫn hơn mọi nhánh học được khi so trên cùng miền AP.

| Task / codec | NPP đối chứng | SAM foreground | SAM foreground + biên | Gaussian σ=0,5 | Gaussian σ=0,8 |
|---|---:|---:|---:|---:|---:|
| IS / H.264 | +5,06% | **−7,52%** | +16,85% | −16,96% | −7,52% |
| IS / H.265 | +3,09% | +7,10% | −0,35% | −2,44% | −6,49% |
| OD / H.264 | +13,74% | +9,99% | +10,10% | −5,23% | −9,73% |
| OD / H.265 | +12,81% | +17,46% | **+32,38%** | +8,07% | −11,71% |

IS là instance segmentation, OD là object detection. Số âm tiết kiệm bitrate tại AP tương đương; số dương là tốn thêm bit so với codec-only. Bảng này dùng BD-rate chuẩn của từng phương án trên miền AP giao với codec-only; các miền giao không hoàn toàn giống nhau.

## Diễn giải

SAM foreground có lợi rõ nhất trong đợt thử này ở IS–H.264: từ +5,06% đối chứng xuống −7,52%. Trên IS–H.265 và OD–H.265, foreground lại làm kết quả xấu hơn đối chứng. Hai nhánh OD vẫn tốn thêm bitrate ở cả hai codec.

Theo BD-rate trên miền giao riêng với codec-only, loss biên xấu hơn foreground ở ba trong bốn cấu hình. Khi dùng cùng miền AP cho tất cả phương án, biên xấu hơn ở hai cấu hình và có lợi ở hai cấu hình; riêng OD–H.264 đổi thứ hạng. Vì vậy không thể kết luận loss biên luôn có hại. Trường hợp bất lợi rõ nhất trong cả hai cách tính là OD–H.265: +32,38% theo miền giao riêng và +32,27% theo miền chung. IS–H.265 có giá trị −0,35% nhưng quá gần 0 để kết luận cải thiện đáng tin cậy từ một seed và 64 ảnh.

Gaussian σ=0,8 đạt BD-rate âm ở cả bốn cấu hình. Việc chọn sigma tốt nhất theo chính tập này chỉ là so sánh phát triển, chưa phải kết quả trên một benchmark độc lập.

## So sánh trên cùng miền AP của tất cả phương án

| Task / codec | NPP đối chứng | SAM foreground | SAM + biên | Gaussian σ=0,5 | Gaussian σ=0,8 |
|---|---:|---:|---:|---:|---:|
| IS / H.264 | +4,87% | −4,98% | +16,85% | −11,67% | −3,98% |
| IS / H.265 | +2,51% | +5,92% | −0,35% | −6,34% | −9,02% |
| OD / H.264 | +11,60% | +9,99% | +7,27% | −1,85% | −8,33% |
| OD / H.265 | +12,40% | +17,46% | +32,27% | +8,18% | −13,65% |

Gaussian tốt nhất vẫn hơn mọi nhánh NPP/SAM ở cả bốn cấu hình. Giá trị −16,96% của Gaussian IS–H.264 trong bảng đầu giảm còn −11,67% nếu ép về miền AP chung; không nên coi mức giảm 15% là kết luận không phụ thuộc miền tích phân.

## Kiểm tra tính hợp lệ

- Đã tải báo cáo cuối đúng phiên bản: IS–H.264 v3; ba worker còn lại v2. Tất cả dùng source SHA 9fa98ecc542dc0bc193c38cfa9dcb0b47f4da5cd44d24916815f3d6d964fa69c.
- Cả ba nhánh trong từng worker dùng cùng hash NPP warm-up, proxy đóng băng, trọng số task và cohort. Đối chứng cũ được giữ lại; số bước huấn luyện lại đối chứng bằng 0.
- Cả 20 đường codec-only đi kèm khớp chính xác với đường đối chứng gốc. Đã tính lại 20 BD-rate bằng tích phân PCHIP và kiểm tra chéo bằng tích phân số.
- Toàn bộ 12 nhánh học được có đủ 768 bước và đủ tám điểm đánh giá. Cả gate proxy ban đầu và cuối đều đạt.
- Tám nhánh SAM có thay đổi tham số thực sự; norm thay đổi khoảng 1,26–1,54. Cache SAM có 64 nguồn, trung bình vùng foreground khoảng 21,46% ảnh và vùng biên khoảng 1,44%; bốn nguồn không có foreground được chấp nhận.
- PSNR NPP trước codec trên dev ở QP35 nằm khoảng 24,58–27,95 dB cho các nhánh SAM. Đây không phải PSNR ảnh giải mã của codec.
- Không sửa kiến trúc NPP và không khởi chạy thử nghiệm mới trong lần kiểm tra này. Tracking vẫn hoãn.

## Phạm vi

COCO2017: 64 nguồn train, 8 nguồn train-dev tách biệt, 64 ảnh VAL; QP 30/35/40/45; mỗi ảnh mã hóa thành một luồng H.264/H.265 độc lập. Đây là thử nghiệm ngắn 768 bước task, một seed, chưa phải toàn bộ lịch huấn luyện của bài báo, chưa đánh giá full COCO 5.000 ảnh và chưa có khoảng tin cậy bootstrap.

Dữ liệu chi tiết, miền AP của từng đường và hash các báo cáo nằm trong RESULTS_20261102.json cùng thư mục.
