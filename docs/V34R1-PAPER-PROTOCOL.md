# Đề xuất thí nghiệm cho bài báo v34r1

Khuyến nghị lấy **một phương pháp tiền xử lý thích ứng theo tác vụ, đánh giá riêng trên ba tác vụ** làm thí nghiệm chính. Ba job chạy tuần tự hay đồng thời không thay đổi câu hỏi khoa học. Cần mô tả rõ phần cốt lõi dùng chung và phần đặc thù AR/OD/OT; hiện critic học rủi ro OD và guard OT khác nhau, nên chưa thể gọi là một mô hình học thống nhất.

Mục tiêu BD-rate <−15% là tiêu chí nghiên cứu, chưa phải kết quả đạt được. Không gộp các codec/task để che một ô chưa đạt. Các phép đo trên calibration đã xem nhiều lần chỉ là phát triển, không phải TEST độc lập.

## Ma trận thí nghiệm chính

| Tác vụ | Dữ liệu hiện tại | Chất lượng chính | Codec | QP |
|---|---|---|---|---|
| AR | Kinetics; công bố đúng tập con và ánh xạ nhãn | Top-1 | H.264, H.265 | 30,35,40,45 |
| OD | COCO2017; ID đã khóa | COCO AP@[.50:.95] | H.264, H.265 | 30,35,40,45 |
| OT | MOT; tách theo video vật lý, đúng FPS gốc | HOTA | H.264, H.265 | 30,35,40,45 |

Bảng chính cần sáu ô BD-rate cùng sáu đường so sánh bitrate–chất lượng, số mẫu và khoảng hỗ trợ chất lượng chung. Giữ cả bốn điểm thô. Khi đường không đơn điệu hoặc không đủ khoảng chung, báo không xác định; không bỏ điểm, chọn lại QP hay sửa đường để đạt−15%. Cỡ mẫu đang dùng để sàng lọc không tự trở thành một benchmark đầy đủ.

## Đối chứng và kiểm chứng khối

So sánh codec gốc/identity, phiên bản trước, cùng ngân hàng ứng viên với bộ chọn cũ, rồi phương pháp mới. Dùng chung detector/recognizer/tracker đóng băng trong từng phép so sánh, cùng nguồn, geometry, codec version/preset/GOP/chroma, QP và quy tắc tính byte. AR/OT là video liên khung; COCO hiện là ảnh mã hóa intra, phải trình bày riêng cấu hình thay vì nói ba tác vụ có cùng GOP.

Ablation lần lượt kiểm tra ngân hàng bộ lọc, vùng bảo vệ đối tượng, proxy tổn hại, critic/bộ chọn và thành phần thời gian của OT. Chỉ so các nhánh thay đúng thành phần được nghiên cứu. GT chỉ dùng huấn luyện/chọn trên FIT/CAL được khai báo và đánh giá; GT oracle là chẩn đoán riêng, không phải phương pháp triển khai.

Kiểm chứng gồm byte stream thực có header, giải mã độc lập, hash nguồn/cấu hình/mô hình, tính đủ lưới bốn QP, tính đúng metric chính thức và trạng thái tracker nhân bản cùng quá khứ/chỉ commit một ứng viên. Báo runtime, GPU/CPU/RAM và số lần encode/decode/suy luận thật. Bộ chọn thử21 bộ lọc phải khai báo chi phí tìm kiếm đó; bitrate truyền thấp không đồng nghĩa encoder rẻ.

## Phát triển và TEST

Khóa seed, dữ liệu, đặc trưng, mô hình, ngưỡng, cách phá hòa, phép nội suy và tiêu chí lỗi trước TEST. Tách theo video cho AR/OT, tránh nhiều cửa sổ cùng video bị tính là các mẫu độc lập. Với OD lấy ảnh làm đơn vị lấy mẫu; với AR/OT lấy video làm đơn vị khi lượng hóa bất định. Bootstrap phải tính lại metric/BD từ mẫu được lấy, đồng thời công bố tỷ lệ lần lấy mẫu có BD không xác định; không chỉ giữ các lần hợp lệ để tạo khoảng tin cậy đẹp.

Các lần chạy lại trên cùng128 video AR chỉ chứng minh tính lặp lại, không tạo thêm dữ liệu độc lập. Hai video FIT hay calibration OT chỉ là thí nghiệm nhỏ, cần thêm dữ liệu độc lập trước khi khái quát. Khi điều chỉnh phương pháp sau khi xem DEV mới, tập đó trở thành dữ liệu phát triển; cần một tập xác nhận chưa xem khác. Không chọn phiên bản theo TEST.

## Nếu muốn tuyên bố phục vụ đồng thời ba tác vụ

Cần video có đủ nhãn AR/OD/OT, một lựa chọn tiền xử lý, một lần mã hóa cho mỗi QP và **cùng bitstream/cùng bản giải mã** đưa vào ba mô hình. Khóa quy tắc cân bằng ba tổn hại trước đánh giá và so với các bộ chọn chuyên một tác vụ trên cùng dữ liệu. Ba kết quả riêng trên Kinetics/COCO/MOT không chứng minh tuyên bố này. Đây là một thí nghiệm bổ sung có phạm vi riêng, chưa nằm trong bằng chứng đã đo.

## Cơ sở tham khảo

- [MPEG: đánh giá VCM cho một hoặc nhiều tác vụ](https://www.mpeg.org/meetings/mpeg-134/). Điều này hỗ trợ việc nêu rõ phạm vi; cấu hình nghiên cứu hiện tại chưa được đối chiếu để tuyên bố tuân thủ CTC MPEG.
- [The Bjøntegaard Bible](https://arxiv.org/abs/2304.12852): phân tích ảnh hưởng của nội suy và cách diễn giải BD; nên báo cả đường cong thay vì chỉ một số tổng hợp. Vẫn giữ đúng bốn QP theo yêu cầu người dùng.
- [HOTA](https://arxiv.org/abs/2009.07736): thước đo tracking cân bằng detection và association. HOTA giữa nguồn và bản giải mã chỉ là proxy; tính đúng nó không chứng minh tương quan với HOTA theo GT hay bảo đảm tiết kiệm bitrate.
