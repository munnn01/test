# Kiểm tra và đánh giá V34r1 A6 — 2026-10-18

**Mục tiêu mới của người dùng: mỗi tác vụ chỉ cần ít nhất một codec có BD-rate hợp lệ < −15%.** AR128, OD192 phát triển và OT hai video đều có ít nhất một điểm BD đạt ngưỡng, nhưng khác phạm vi dữ liệu và chưa hoàn tất đánh giá tăng mẫu. Chỉ dùng QP30/35/40/45. Tất cả mô hình, codec, tracker và chấm GT thực tế chạy ở Kaggle. Tiêu chí một codec được sửa sau khi đã có kết quả A6; các báo cáo A6 gốc và cổng hai codec được giữ nguyên.

| Phép đo | H.264 | H.265 | Phạm vi bằng chứng |
|---|---:|---:|---|
| AR A2 | -31,35% | -21,22% | 128 video đã đánh giá; CI chưa xác nhận ngưỡng |
| OD A6 | -21,15% | -14,30% | Dự đoán chéo trên 192 ảnh phát triển; chưa chạy pilot64 |
| OT A5, tham chiếu lịch sử | -12,94% | -7,67% | Hai video calibration tái sử dụng |
| OT A6, toàn21 cấu hình | không xác định | -16,49% | Cùng hai video calibration; chính sách chọn chỉ từ FIT |

## AR

Hai điểm BD của A2 đạt mục tiêu số học nhưng bootstrap5000 có mẫu không tính được BD, nên giới hạn tin cậy bảo thủ chưa xác nhận ngưỡng. Các lần chạy lặp trước dùng cùng128 video và không làm tăng cỡ mẫu.

Job A6 đã khóa512 ID video độc nhất,8192 khung hình chọn và16384 hash JPEG trong các cửa sổ nguồn. Các hash lịch sử và chỉ số16 khung chọn của128 video cũ khớp. Job này có0 suy luận mô hình và0 thử codec: **chưa có kết quả AR512**. Không gọi khóa dữ liệu là hoàn tất tăng mẫu đánh giá.

## OD

Ba nhóm64 ảnh đã được tiêu thụ cho nghiên cứu tạo thành192 ảnh phát triển. Mỗi lượt giữ nguyên một nhóm64 để dự đoán, chỉ học từ128 ảnh ở hai nhóm còn lại. Nhãn tác động AP được tạo riêng trong từng nhóm.302 đặc trưng phía gửi không có GT, ID ảnh hoặc nhãn nhóm.

Sáu mô hình theo lượt và hai mô hình cuối đã được lưu/hash trước khi chấm các lựa chọn OOF. Tám phép đối chiếu sklearn/JSON có sai số tối đa5,6×10⁻¹⁷;16 phép thay thế nhãn mới khớp COCOeval chính thức với sai số0. Lưới đủ32256 hàng nhưng đơn vị ảnh độc lập chỉ là192, không phải32256.

Quy tắc utility đã đăng ký trước chọn depth6/ngưỡng0, sau đó khóa mô hình cuối học trên192 ảnh. Trên OOF, H.264−21,15% đạt mục tiêu điểm theo tiêu chí mới; H.265−14,30% tính được BD hợp lệ nhưng chưa vượt−15%. Cả14 lựa chọn đều không đạt đồng thời hai codec theo tiêu chí cũ. Chính sách giữ identity ở418/1536 điều kiện ảnh/codec/QP.

**Đánh giá:** có tín hiệu đáng tiếp tục ở OD nhưng đây là kết quả phát triển đã dùng để chọn chính sách, chưa phải xác nhận độc lập. Mô hình triển khai học192 ảnh khác các mô hình OOF học128 ảnh; ngưỡng có thể dịch chuyển. Không so trực tiếp mức cải thiện với A4 fresh64 vì khác tập ảnh. Đo phân biệt tiếp theo là chạy nguyên chính sách đã khóa trên64 ảnh mới; chưa mở OD256.

## OT

Cả bốn job GPU đã hoàn tất: bốn video MOT17,64 khung/video, đủ2688 hàng21 cấu hình×2 codec×4 QP×4 cửa sổ×4 video. Có2696 encode và2696 decode thật, gồm8 lần lặp anchor;43136 suy luận detector trên khung giải mã.512 lượt suy luận nguồn tương ứng256 khung nguồn duy nhất do hai worker codec cùng đọc nguồn.

Cả772 cấu hình thuộc ngân hàng cũ qua kiểm tra byte packet, pixel giải mã và dự đoán detector; tám lần lặp anchor đạt. Các job tạo cache không giải nén GT và không chấm HOTA: **cache hoàn tất chưa chứng minh BD-rate đạt mục tiêu**. Số ứng viên logic tăng khoảng3,48 lần so với772 hàng cũ; phải báo chi phí tìm kiếm này trong bài báo.

Job CPU đã phát lại lịch sử nhân quả, thử bảy lambda chỉ trên FIT và khóa {'kind': 'source_hota_lagrangian', 'weight': 32} trước calibration. Anchor và đối chứng lambda32/shortlist tái lập các tệp tracker và metric cũ. Kết quả A6: không xác định | -16,49%.

H.264 không xác định vì HOTA ở QP30 là53,479675%, thấp hơn QP35 là53,938814%; đảo chiều chủ yếu ở MOT17-11. Cả bốn điểm đều tốt hơn anchor về rate và HOTA, nhưng điều đó không tạo ra BD hợp lệ. Không bỏ điểm hoặc sửa đường cong. A7 sẽ thử dùng chung một profile cho bốn QP, chọn tham số trên FIT rồi khóa trước một lần CAL; chưa có kết quả A7.

Theo tiêu chí một codec được sửa sau A6, cổng tăng mẫu: đủ điều kiện đo tiếp8 video nhờ H.265; chưa phải xác nhận thống kê. Cổng gốc hai codec vẫn được lưu với trạng thái không đạt. Calibration vẫn là hai video đã dùng trong nghiên cứu; không được gọi là dữ liệu xác nhận mới.

## Ý nghĩa cho bài báo và tăng mẫu

- Có điểm BD đạt ngưỡng mới ở cả ba tác vụ trong các phạm vi hiện tại. Chưa thể viết kết luận xác nhận chung hoặc kết quả tăng mẫu. Ghi rõ TEST/OOF phát triển/calibration tái sử dụng cho từng hàng.
- Giữ tất cả bốn điểm thô và các kết quả âm; không ghép cấu hình tốt nhất riêng cho từng codec sau khi xem kết quả.
- AR512 đã có nguồn khóa nhưng chưa đánh giá. OD cần pilot64 mới trước quyết định OD256. OT8 đủ điều kiện theo cổng đã sửa nhưng chưa chạy. Đạt điểm BD chưa tự xác nhận khoảng tin cậy.
- OT8 đã dành năm video MOT17 và ba video DanceTrack0081/0090/0035. Đã loại MOT20-02/05/03 vì chúng từng là train/cal/dev. Hai video OT cũ phải báo riêng với sáu video bổ sung.

## Bằng chứng

- [OD: số bốn QP](V34R1-A6-OD-OOF-CURVES.csv) và [đủ14 cấu hình](V34R1-A6-OD-ALL14.csv).
- [Đầu ra OD đã khóa trên Git](https://github.com/munnn01/test/tree/c46327828ab252ce81dc8501d7202ced49908046/results/v34r1_a6/od-crossfit).
- [Bằng chứng khóa nguồn AR512](https://github.com/munnn01/test/tree/c46327828ab252ce81dc8501d7202ced49908046/results/v34r1_a6/ar512-sources).
- [Kiểm toán cache OT](https://github.com/munnn01/test/tree/c46327828ab252ce81dc8501d7202ced49908046/results/v34r1_a6/ot-cache-audit).
- [Bốn điểm thô OT A6](V34R1-A6-OT-CURVES.csv) và [bằng chứng OT A6](V34R1-A6-OT-EVIDENCE.tgz).

Kiểm tra trên laptop chỉ đọc đầu ra, kiểm tra lưới/hash/byte và trình bày số đã đo. Không chạy đặc trưng, mô hình, codec, tracker hoặc GT thật trên máy.
