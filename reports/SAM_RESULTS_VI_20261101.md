# Kết quả kiểm tra ngày 01/11/2026

NPP đối chứng chưa tiết kiệm bitrate. Gaussian σ=0,5 tốt hơn NPP trên cả bốn đường đã đánh giá; riêng IS–H.264 đạt BD-rate −16,96%. Hai nhánh SAM chưa có kết quả RD hoàn chỉnh trong snapshot này.

| Task / codec | NPP đối chứng | Gaussian σ=0,5 |
|---|---:|---:|
| Instance segmentation / H.264 | +5,06% | −16,96% |
| Instance segmentation / H.265 | +3,09% | −2,44% |
| Object detection / H.264 | +13,74% | −5,23% |
| Object detection / H.265 | +12,81% | +8,07% |

BD-rate âm là tiết kiệm bit tại AP tương đương; dương là tốn thêm bit. Đánh giá trên cùng 64 ảnh COCO VAL, QP 30/35/40/45, mỗi ảnh là một luồng H.264/H.265 độc lập. Đây chưa phải đánh giá full COCO hoặc video có dự đoán liên khung. Các đường Gaussian được tính từ đủ tám điểm trong log đang chạy; cả bốn điểm codec-only khớp chính xác với đối chứng gốc.

Cả bốn proxy đều vượt gate trước và sau 768 bước huấn luyện NPP. Tuy vậy, NPP vẫn thua baseline Gaussian; proxy đạt gate không đảm bảo tối ưu NPP thành công. Log gradient cho thấy tỷ lệ norm rate/task ở QP45 chỉ khoảng 0,023–0,048, so với 2,10–5,23 tại QP30. Đây là dấu hiệu cần khảo sát cân bằng mục tiêu theo QP, chưa phải bằng chứng nhân quả.

Bốn run gốc v1 dừng khi đánh giá Gaussian do StopIteration: bộ lọc chỉ có buffer, trong khi processed_video lấy thiết bị từ parameter. Đã sửa bằng một parameter cố định, kiểm tra lại 196 pytest và 24 điểm tích hợp codec thật. Giữ nguyên NPP tác giả và bốn QP; không thêm identity/residual.

Đã phục hồi trên bốn tài khoản bằng bản sao CPU riêng tư v2 của đúng artifact v1. Phiên bản v2 này chỉ sao chép artifact, không huấn luyện. Tám file checkpoint tải về local khớp SHA256 với bản sao trên cloud. Không chạy lại nhánh đối chứng; hai nhánh SAM dùng lại warm-up, proxy, optimizer và RNG đã lưu. Nếu đường codec-only không khớp, phục hồi sẽ dừng.

Bốn worker phục hồi đã được xác nhận RUNNING, notebook riêng tư và đầu vào gắn kèm đúng. Kết quả SAM và Gaussian σ=0,8 chưa được chốt trong báo cáo này. Tracking vẫn được hoãn.

Code: https://github.com/munnn01/test/commit/d7a9ad7bfe8746cb38489b9998f5f736984b9d46
