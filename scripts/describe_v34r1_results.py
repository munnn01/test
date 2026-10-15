"""Export measured tables and a Vietnamese evidence report, without task inference."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
report=json.loads((args.root/'target-report.json').read_text())
metrics={'ar':'top1','od':'COCO_AP','ot':'HOTA_div_100'}
curve_rows=[]; endpoint_rows=[]; cost_rows=[]; runs={}
for task in ('ar','od','ot'):
    root=args.root/(task+'-test')
    summary=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    rows=json.loads(gzip.decompress((root/'rows.json.gz').read_bytes()))
    runs[task]=dict(summary=summary,protocol=protocol)
    for codec in ('h264','h265'):
        for arm in ('anchor','adaptive'):
            curve=summary['curves'][codec][arm]
            for qp,rate,quality in zip(protocol['config']['qps'],curve['rate'],curve['quality']):
                curve_rows.append(dict(task=task,codec=codec,arm=arm,qp=qp,packet_bits_per_canonical_pixel=rate,
                    quality_metric=metrics[task],quality=quality,source_or_window_count=summary['grid_audit']['source_count']))
        result=summary['comparisons'][codec]['adaptive']; ci=result.get('ci') or {}
        endpoint_rows.append(dict(task=task,codec=codec,bd_rate_pct=result['pchip_bd_rate_pct'],valid=result['valid'],
            point_passes=result['point_passes'],threshold_confirmed=result['threshold_confirmed'],
            invalid_reason=result.get('invalid_reason'),ci_lo=ci.get('lo'),ci_hi=ci.get('hi'),
            bootstrap_finite_fraction=ci.get('finite_fraction')))
        adaptive=[r for r in rows if r['codec']==codec and r['arm']=='adaptive']
        anchor={(r['id'],r['qp']):r for r in rows if r['codec']==codec and r['arm']=='anchor'}
        if not all(r['bytes']<=anchor[(r['id'],r['qp'])]['bytes'] for r in adaptive): raise ValueError('byte budget violated')
        cost_rows.append(dict(task=task,codec=codec,conditions=len(adaptive),
            mean_selected_to_anchor_byte_ratio=statistics.mean(r['bytes']/anchor[(r['id'],r['qp'])]['bytes'] for r in adaptive),
            identity_fraction=statistics.mean(r['action']==0 for r in adaptive),
            unlearned_profile_selected_fraction=statistics.mean(r['action']>=13 for r in adaptive),
            mean_distinct_trial_encodes=statistics.mean(r['candidate_encode_count'] for r in adaptive),
            mean_sender_selection_seconds=statistics.mean(r['sender_selection_seconds'] for r in adaptive),
            mean_receiver_all_arms_seconds=statistics.mean(r['receiver_all_arms_seconds'] for r in adaptive),
            whole_task_benchmark_wall_seconds=summary['runtime']['wall_seconds'],
            whole_task_peak_torch_cuda_allocated_bytes=summary['runtime']['peak_torch_cuda_allocated_bytes']))
for name,rows in [('V34R1-TEST-CURVES.csv',curve_rows),('V34R1-ENDPOINTS.csv',endpoint_rows),('V34R1-COSTS.csv',cost_rows)]:
    with (args.out/name).open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
def value(result):
    return f"{result['pchip_bd_rate_pct']:.2f}%" if result['valid'] else 'Không hợp lệ: '+result['invalid_reason']
lines=['# Kết quả kiểm chứng v34r1','',
    '**Chưa đạt mục tiêu BD-rate < −15% đồng thời cho AR, OD và OT trên cả hai codec.**' if not report['all_point_targets'] else
    '**Sáu ước lượng điểm đều nhỏ hơn −15%; xem riêng điều kiện xác nhận CI bên dưới.**','',
    '| Tác vụ | H.264 | H.265 |','|---|---:|---:|']
for task in ('ar','od','ot'):
    r=runs[task]['summary']['comparisons']
    lines.append('| '+task.upper()+' | '+value(r['h264']['adaptive'])+' | '+value(r['h265']['adaptive'])+' |')
lines += ['',f"`all_point_targets={str(report['all_point_targets']).lower()}`; `all_thresholds_confirmed={str(report['all_thresholds_confirmed']).lower()}`.",'',
    'BD-rate dùng PCHIP của log-rate trên khoảng chất lượng chung; chỉ QP30/35/40/45, không ngoại suy hoặc sửa đường. Giá trị âm là tiết kiệm byte tại cùng chất lượng tác vụ. Mọi endpoint không hợp lệ vẫn xuất hiện trong bảng và ngăn kết luận đạt đồng thời.','',
    '![Bốn điểm đo thực tế cho từng tác vụ và codec](figures/v34r1_test_curves.png)','',
    'Hình nối các điểm thực đo, không làm trơn hay sửa đường. Trục x là tổng bit packet trên pixel RGB canonical; trục y lần lượt là top-1, COCO AP và HOTA/100. CSV giữ đủ 48 điểm.','',
    '## Bằng chứng thực thi','',
    'Toàn bộ TEST cuối chạy trên Kaggle, GPU T4×2, cùng Docker image và FFmpeg Linux7.0.2 đã ghim hash. Laptop chỉ điều phối, sao chép, kiểm toán và vẽ dữ liệu đã đo; các TEST Windows dở dang bị loại, không trộn vào đường cuối.','',
    '| Tác vụ | Nguồn giữ lại | Hàng ghép cặp | Thời gian đo, giây | Kiểm thử trên Kaggle |','|---|---|---:|---:|---:|']
scopes={'ar':'128 video;16 frame/video','od':'64 ảnh COCO2017 val','ot':'1 sequence MOT20-01;256 frame;16 cửa sổ'}
for task in ('ar','od','ot'):
    summary=runs[task]['summary']; audit=json.loads((args.root/(task+'-test')/'cloud/import-audit.json').read_text())
    lines.append(f"| {task.upper()} | {scopes[task]} | {summary['grid_audit']['paired_rows']} | {summary['runtime']['wall_seconds']:.1f} | {audit['block_tests_passed']} passed |")
lines += ['',
    'Thời gian đo gồm chuẩn bị nguồn tại sender, tìm ứng viên, mọi lần encode/decode thực và scoring; không gồm tải dữ liệu/trọng số và khởi tạo model. File COSTS ghi trial encode, tỷ lệ identity/profile cố định và thời gian; peak CUDA chỉ là số byte PyTorch cấp phát, không phải toàn bộ bộ nhớ GPU.','',
    'Chính sách A2 được đóng băng trước TEST. Namespace thuật toán và configs giữ nguyên giữa commit 5a36901 và bản sửa vận chuyển nguồn b3f30bc; log/protocol lưu hash thực tế để kiểm tra chéo. Các packet được chọn trong TEST đều dùng identity hoặc profile cố định mới; controller được giữ đóng băng.','',
    'AR chuyển từ mount provider 23 GB và tải lẻ lỗi HTTP sang bộ đệm Kaggle riêng tư gồm 3.053 JPEG, 43,8 MB. Từng byte khớp manifest đã đóng băng; ID, frame và label giữ nguyên. Archive nguồn SHA-256: `727e1b582943e77cdf509b0a27986af52bdb13a7d6dd046de2ad72fc03caac19`. Git chỉ chứa metadata/hash, mã và kết quả, không chứa media nguồn hoặc thông tin đăng nhập.','',
    '## Độ tin cậy và giới hạn','',
    'AR bootstrap ghép cặp theo video, 5.000 lần; draw không hợp lệ được trả vào đuôi bất lợi không hữu hạn. CI một phía điều chỉnh Bonferroni cho sáu endpoint. OD chưa có bootstrap AP đầy đủ, và OT chỉ có một sequence nên không thể xác nhận hiệu quả ở quần thể sequence. Kiểm thử mã và bằng chứng hiệu năng được đánh giá riêng.','',
    '| AR endpoint | CI lower | CI upper | Tỷ lệ draw hợp lệ |','|---|---:|---:|---:|']
for codec in ('h264','h265'):
    ci=runs['ar']['summary']['comparisons'][codec]['adaptive'].get('ci') or {}
    bound=lambda name: f"{ci[name]:.2f}%" if ci.get(name) is not None else 'Không hữu hạn'
    lines.append(f"| {codec.upper()} | {bound('lo')} | {bound('hi')} | {ci.get('finite_fraction',0):.4f} |")
lines += ['',
    'K400 TEST dùng128 ID official-val mới, loại holdout/controller/exclusions lịch sử; cadence gốc của JPEG provider chưa xác minh. COCO64 là cohort hữu hạn. MOT20-01 là holdout nội bộ từ official train, không phải bảng điểm challenge test. Không suy rộng kết quả này cho toàn bộ K400/COCO/MOT20.','',
    'Guard dùng dự đoán nguồn sạch và có thể giữ lỗi của model nguồn. AR top-1, COCO AP và HOTA đều được chấm bằng GT sau lựa chọn packet. Các profile mới là phép lọc cố định, không phải đầu ra mới của controller được huấn luyện.','',
    'AutoResearchClaw qua ACP đã phản biện thiết kế thật; CLI research đầy đủ trước đó dừng ở TOPIC_INIT. Phản biện và bài báo tạo giả thuyết, không thay thế các số liệu TEST trên.','',
    '## Hồ sơ tái lập','',
    '- [Mã và nhánh v34r1](https://github.com/munnn01/test/tree/v34r1).',
    '- `results/v34r1`: protocol, paired rows nén, summary, tracking predictions, manifest hash, target report và log/notebook Kaggle.',
    '- `scripts/verify_v34r1_results.py`: kiểm toán CPU hash, lưới, đường AR và phép tính BD-rate; không tuyên bố chạy lại detector hoặc chấm lại COCO/TrackEval.',
    '- `V34R1-TEST-CURVES.csv`, `V34R1-ENDPOINTS.csv`, `V34R1-COSTS.csv`: bảng dữ liệu của báo cáo.',
    '- Các DEV exploratory và thất bại được giữ riêng; không đổi chính sách theo TEST để trình bày một lượt đạt giả.','']
(args.out/'V34R1-REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(curve_points=len(curve_rows),endpoints=len(endpoint_rows),all_point_targets=report['all_point_targets'])))
