# AR: kiểm tra lặp lại phép chạy A2

Hai job nguồn từng chờ tải dữ liệu đã hoàn tất. Cả hai khớp lượt chính về đường cong rate/accuracy, toàn bộ kết quả so sánh/CI đã tính, hash manifest nguồn, hash task model, hash controller và đúng128 ID video. H.264 **−31.35066760285326%**, H.265 **−21.221701678883925%**; đúng QP30/35/40/45.

Đây là lặp lại tính toán trên **cùng128 video**. Không cộng thành384 video độc lập, không chọn lượt có kết quả đẹp hơn, không thay lượt chính đã công bố. Chưa so lại toàn bộ packet của các lượt bổ sung nên không gọi đây là chứng minh mọi packet giống byte. Khoảng tin cậy AR trong báo cáo chính vẫn chưa xác nhận ngưỡng; các giới hạn về cadence của JPEG nguồn cũng giữ nguyên.

| Lượt | Commit cloud | Phạm vi |
|---|---|---|
| Chính đã công bố | b3f30bc | [AR cache TEST](https://www.kaggle.com/code/baooo25r/v34r1-ar-cache-test-qp4) |
| Nguồn mounted bổ sung |5a36901| [AR mounted TEST](https://www.kaggle.com/code/shungg05/v34r1-ar-test-qp4) |
| HTTP bổ sung |b3f30bc| [AR direct TEST](https://www.kaggle.com/code/baooo25r/v34r1-ar-direct-test-qp4) |

Các commit5a36901 vàb3f30bc có cùng mã/chính sách khoa học; b3 chỉ sửa retry khi truyền nguồn. Evidence tại `results/v34r1_a3/ar-repeatability/` gồm summary/protocol/status của hai lượt và bảng đối chiếu. Các job đều đã hoàn tất; không có job AR thừa đang giữ trạng thái chạy ở lần kiểm tra này.
