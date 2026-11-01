# Phân tích và đợt thử H.264/H.265 — 01/11/2026

Kết quả BPG trước: IS không SAM −9,81%, IS có SAM +7,69%; OD không SAM +2,74%, OD có SAM −12,98%. Đây là 32 ảnh VAL, 256 bước task, chưa phải benchmark đầy đủ. Không chuyển các con số này thành kết quả H.264/H.265.

## Điều đã quan sát được

- IS: SAM giảm mask AP so với đối chứng tại QP30: 41,51→39,95; QP35:37,30→34,50; QP40:32,11→29,88, trong khi bpp gần bằng hoặc tăng. QP45 SAM tốt hơn:21,01→22,60 AP. Hiệu quả phụ thuộc mức nén, không phải SAM thất bại ở mọi QP.
- OD: tại QP40 SAM tăng AP31,54→37,75 nhưng tăng bpp0,0941→0,1001; tại QP45 tăng18,95→24,45 nhưng bpp0,0525→0,0562. Đây là đổi bit lấy chi tiết hữu ích; BD-rate mới phản ánh hiệu quả tại cùng AP.
- SAM tạo mask không rỗng trên60/64 ảnh TRAIN, coverage trung bình21,46%; vấn đề không phải toàn bộ cache mask rỗng.
- IS có SAM đạt PSNR trước codec25,19dB ở probe task cuối, cao hơn đối chứng22,96dB, nhưng BD-rate kém hơn. Bảo toàn pixel trung bình không bảo đảm giữ đúng đặc trưng mask.
- Warm-up IS3000/4000 bước khác nhau; OD cùng3000 nhưng trạng thái khác nhau. Kết quả chưa tách được tác động nhân quả riêng của SAM.

## Giả thuyết từ code cần kiểm chứng

1. Mask SAM là union mềm của tối đa8 vùng, đã làm mượt5x5. Dpre tập trung vào interior/objectness, không có mục tiêu bảo vệ biên riêng của từng instance. Phạt cao tần nền có thể làm mất vật nhỏ, biên và ngữ cảnh. Đây là suy luận cơ học; chưa có đo phổ/biên để khẳng định là nguyên nhân chính.
2. Mask R-CNN trả nhiều loss RPN, classification, regression và mask. Adapter cũ cộng tất cả bằng trọng số1, không ghi từng thành phần, nên chưa biết gradient mask có bị lấn át không. Nguồn: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html
3. Chỉ256 cập nhật trên64 ảnh dễ chưa hội tụ và nhạy với mẫu/QP. LR1e-4 có thể khiến ảnh rời xa warm-up; loss/train giảm chưa chứng minh VAL tốt hơn. Cần loss theo QP và đánh giá độc lập.
4. Proxy cũ được fit cho BPG. H.264/H.265 có tái tạo, chuyển màu420 và chi phí header khác nhau, nên cần refit riêng và đo hướng gradient codec thật. Phép probe mới vẫn cục bộ trên TRAIN-dev gốc, không chứng minh toàn bộ phân phối đầu ra NPP.

## Thử nghiệm triển khai

Giữ NPP Conv1x1+U-Net/QP → codec thật → task đóng băng, proxy backward, SAM offline. Không identity. Mỗi worker chạy hai nhánh liên tiếp, dùng đúng cùng snapshot warm-up/proxy/optimizer/RNG. Chỉ chấp nhận warm-start khi source/config/cohort khớp, task_updates=0 và warm-up hội tụ. Không chuyển checkpoint giữa các tài khoản.

IS: cả hai nhánh dùng loss_mask×3; nhánh cải tiến dùng foreground_extra1 thay3, bỏ gamma nền, thêm loss tái tạo ở biên riêng từng instance GT TRAIN với trọng số2. Biên được tính từng instance trước khi hợp nhất để giữ biên vật chạm nhau. GT này không được đưa vào suy luận/đánh giá. Đây là can thiệp kết hợp, không phải ablation riêng SAM.

OD: nhánh cải tiến giảm foreground_extra3→1,5 và gamma0,1→0,02; đối chứng không SAM. Cả hai nhánh dùng768 task updates, LR3e-5 và luân phiên đều4QP. Mỗi codec refit proxy256 bước trên TRAIN, đo rate/reconstruction trên TRAIN-dev và hướng gradient rate bằng perturb±2/255 với codec thật.

Đánh giá64 ảnh COCO VAL, AP mask/bbox,8 điểm (2 đường×4QP) mỗi nhánh. Mỗi codec có baseline riêng; tính BD-rate PCHIP chỉ trong miền AP giao nhau, không ngoại suy và không ép kết quả khi đường không đơn điệu. 64 ảnh vẫn là pilot; cần xác nhận tập lớn/seeds mới kết luận đạt−15%.

COCO ảnh tĩnh được mã hóa thành stream1frame độc lập bằng libx264/libx265, yuv420p, presetmedium, −qp30/35/40/45; bpp tính mọi byte stream gồm header. Không ghép ảnh không liên quan thành video để tạo lợi ích temporal giả. Nguồn option QP: https://github.com/FFmpeg/FFmpeg/blob/master/doc/encoders.texi

4worker trên4tài khoản;6000s/arm, timeout13200s/worker, tổng trầnworker13,33h và reservation14,67h. Quota được kiểm tra trước khi submit. Bảo lưu gate25dB warm-up và20dB task; nếu không vượt thì báo chưa train, không fallback. Tracking vẫn chưa khởi chạy.
