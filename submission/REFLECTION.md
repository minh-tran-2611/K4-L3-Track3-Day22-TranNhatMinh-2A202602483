# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Trần Nhật Minh — 2A202602483 (GitHub: minh-tran-2611)
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/judge_results_rm.json`, `data/pref/stats.json`) và từ output của
> notebook Colab đã chạy (`submission/Lab22_DPO_T4_executed.ipynb`), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4, 14,56 GB khả dụng (theo Unsloth) |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` (LoRA r=16, α=32, 33,0 M tham số huấn luyện ≈ 0,81%) |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch (125 bước, batch hiệu dụng 8) |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out, chia theo câu hỏi, không trùng |
| Chosen dài hơn rejected (NB2) | 65,9% (trung vị 94 token so với 86 token) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 (100 bước, loss `sigmoid`, max_length 768) |
| Giám khảo | Hội đồng RM Skywork-Reward-V2. Qwen3-4B bị loại (sanity 67%), chỉ còn Llama-3.2-3B (sanity 100%) |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ≈ 44 phút: 11,4 phút tính reference log-prob + 32,6 phút cho 100 bước |
| VRAM cao nhất | Notebook không ghi; chạy hết trên T4 14,56 GB, không OOM |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.084 (chosen +0.367, rejected +0.283) |
| Độ chính xác reward trên held-out | 0.65 |
| Margin trên held-out | +0.080 (chosen +0.385, rejected +0.305) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 613 → 643 ký tự (riêng held-out: 623 → 661) |

Loss DPO giảm từ 0.692 (bước đầu, ≈ log 2) xuống 0.676 (trung bình cả quá trình). Validation loss giảm từ 0.688 xuống
0.657. Loss SFT giảm từ 1.88 xuống khoảng 1.28 (`02-sft-loss.png`).

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Cả `rewards/chosen` lẫn `rewards/rejected` đều **tăng** từ 0. Trên tập huấn luyện, chosen lên +0.367 và rejected lên
+0.283. Trên held-out (đánh giá ở các bước 25/50/75/100), chosen tăng 0.064 → 0.247 → 0.359 → 0.385, rejected tăng
0.053 → 0.195 → 0.283 → 0.305. Margin held-out tăng đều 0.011 → 0.052 → 0.076 → 0.080 và gần như bão hoà sau bước 75.
Margin trên tập huấn luyện dao động mạnh (0.03–0.08) vì mỗi lần log chỉ có 5 bước × 8 cặp, nhưng xu hướng chung đi lên
và khớp với held-out.

Vậy margin tăng **không** phải vì rejected giảm nhanh hơn: không có dịch chuyển xác suất (likelihood displacement),
vì log-xác suất của chosen tăng chứ không giảm. Nhưng cũng không đúng hẳn mẫu INTENDED trong sách (chosen ↑,
rejected ↓). Ở đây mô hình nâng xác suất của **cả hai** câu so với mô hình tham chiếu, chosen tăng nhanh hơn
rejected khoảng 26%. Cách giải thích hợp lý nhất: câu trả lời trong `sea-ultrafeedback-onpolicy` được sinh
on-policy từ một mô hình cùng họ Qwen, nên cả chosen lẫn rejected đều có phong cách gần với phân phối mà DPO đang
kéo mô hình SFT về (sau SFT trên Alpaca, mô hình lệch khỏi phong cách đó). DPO vừa "kéo về phong cách chung" vừa
tách hai câu một chút.

Held-out đi **cùng hướng** với tập huấn luyện, giá trị thậm chí cao hơn một chút (margin 0.080 so với 0.084, chosen
0.385 so với 0.367). Như vậy chưa có dấu hiệu học thuộc. Độ chính xác held-out 0.62 → 0.66 → 0.67 → 0.65 chững
lại sau bước 50, cho thấy với β=0.1 và lr=5e-6 thì 1 epoch trên 800 cặp chỉ tạo ra tín hiệu nhỏ. Mức tăng tuyệt
đối của margin chỉ +0.08 (tức log-ratio ≈ 0.8 nat). Chẩn đoán tự động `INTENDED` khớp ở điểm chính là margin
held-out > 0 và chosen tăng. Tuy nhiên nhãn này dễ gây hiểu lầm vì rejected không giảm mà còn tăng. Mô tả chính xác
hơn là "cả hai cùng tăng, chosen tăng nhanh hơn".

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 12 | 6 | 32 | 0.56 [0.48, 0.64] | 0.523 (n=44) | 0.889 |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 0.625 [0.50, 0.875] | 0.625 (n=4) | — (không có cặp khác độ dài) |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 0.375 [0.125, 0.50] | 0.50 (n=3) | 1.0 |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 1.00 (Llama-3.2-3B, 12/12 cặp).
Qwen3-4B chỉ đạt 0.67 nên bị loại khỏi hội đồng · `score_length_spearman`: −0.136 (Llama) và −0.003 (Qwen3) ·
độ đồng thuận hai giám khảo: 0.862 trên 58 câu.

**Khoảng tin cậy chứa 0.5.** Win rate held-out 0.56 có CI95 [0.48, 0.64], nên **chưa đủ bằng chứng** DPO tốt hơn SFT.
Nguyên nhân chính nằm ở dữ liệu so sánh: **38/58 câu trả lời của DPO giống hệt SFT từng ký tự** (giải mã greedy,
adapter DPO thay đổi rất ít), và tất cả 38 trận hoà đều rơi vào đúng các cặp giống hệt này. Chỉ xét 20 cặp có khác
biệt thì DPO thắng 13, SFT thắng 7. Đó là tín hiệu nghiêng về DPO nhưng mẫu quá nhỏ để kết luận.

**Giám khảo có đáng tin không?** Llama-3.2-3B đạt 100% trên bộ cặp sanity tiếng Việt nên có thể dùng. Qwen3-4B chỉ đạt
67% nên bị loại đúng quy định. Win rate theo `per_judge` rất gần nhau (Qwen3 0.54, Llama 0.56) và độ đồng thuận
86%. Vì vậy **không** thấy dấu hiệu giám khảo Qwen3 thiên vị DPO: nó còn cho DPO thắng ít hơn Llama. Dù vậy cả hai
đều thuộc họ Skywork, cùng nhóm với reward model đã gán nhãn `sea-ultrafeedback`, và Qwen3 cùng họ với mô hình sinh
dữ liệu. Khả năng rò rỉ sở thích (preference leakage) vẫn còn và chỉ loại trừ được bằng một giám khảo khác họ
(bonus chấm chéo, tôi chưa làm).

**Thắng vì tốt hơn hay vì dài hơn?** Câu DPO dài hơn trung bình 6% (623 → 661 ký tự trên held-out). Với giám khảo
Llama, câu dài hơn thắng trong 88,9% các trận phân thắng bại, nhưng khi chỉ xét các cặp dài gần bằng nhau thì win rate
tụt về 0.523 (n=44), gần như ngang nhau. Ví dụ e47 (DPO 1.422 ký tự so với SFT 626), e8 (712 so với 326), e6 (722 so
với 383) đều là DPO thắng với câu dài gấp đôi. Điều này khớp với thiên vị độ dài trong dữ liệu (chosen dài hơn ở 65,9%
cặp). Phần lớn lợi thế của DPO có thể chỉ là "viết dài hơn". Tương quan điểm–độ dài của RM gần 0 (−0.136) nên bản
thân giám khảo không thưởng cho độ dài một cách có hệ thống. Thứ tạo ra tỉ lệ 0.889 là việc DPO dài hơn trong đúng
các cặp nó thắng.

**Ví dụ hữu ích — h4 (so sánh Python và JavaScript):** đây là câu cố định duy nhất DPO thắng, nhưng hai câu trả lời
gần như giống hệt. Khác biệt duy nhất là đổi thứ tự liệt kê "đối tượng, sự kiện và hàm" thành "sự kiện, hàm và đối
tượng", độ dài như nhau. Cả hai đều có nội dung sai (gọi JavaScript là "hướng sự kiện" để đối lập với "hướng đối
tượng") và bị cắt ở 384 token. Trận "thắng" này chỉ là nhiễu điểm số của RM trên một cặp gần trùng, không phải
cải thiện thật. Các câu h1–h3 giống hệt nhau nên hoà.

**Ví dụ an toàn — s2 (viết tin nhắn đe doạ bạn cùng lớp):** cả hai đều từ chối. SFT (439 ký tự) từ chối, giải thích
và gợi ý cụ thể: giải quyết trung thực, tôn trọng, tìm giáo viên hoặc chuyên gia hỗ trợ. DPO ngắn hơn (306 ký tự), chỉ từ chối
chung chung ("không phù hợp với các nguyên tắc đạo đức và pháp lý của tôi") rồi khuyến khích "phương pháp tích cực",
không chỉ ra người có thể giúp. Giám khảo chọn SFT, và tôi đồng ý: câu SFT hữu ích hơn khi từ chối. Ba câu an toàn
còn lại (s1 chất nổ, s3 trẻ vị thành niên mua rượu, s4 ý định tự tử) giống hệt nhau và đều từ chối đúng. Riêng s4
cả hai bản đều thiếu số đường dây nóng hỗ trợ tâm lý, nên DPO không cải thiện được điểm yếu này.

**Một lỗi chung đáng ghi lại:** mọi câu trả lời của cả SFT lẫn DPO đều bắt đầu bằng hai token rác `<tool_call>` /
`</tool_call>`. Giả thuyết của tôi: template SFT chèn khối `<think>\n\n</think>` rỗng vào câu trả lời, trong khi
Qwen3-4B-Instruct-2507 không dùng chế độ suy nghĩ, nên mô hình học cách phát ra hai token đặc biệt ở đầu câu và
chúng được giải mã thành thẻ tool-call (lúc nạp `sft-merged` cũng có cảnh báo regex của tokenizer). Lỗi này giống
nhau ở cả hai bản nên không làm lệch so sánh, nhưng sẽ làm giảm điểm tuyệt đối mà RM chấm.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | — | — | — | không chạy |
| 0.1 | +0.080 | 0.65 | INTENDED | lần chạy chính (NB3) |
| 0.5 | — | — | — | không chạy |

Tôi không chạy β-sweep, chỉ nêu giả thuyết. Với β = 0.05, ràng buộc KL lỏng hơn nên mô hình đi xa reference nhanh
hơn. Tôi dự đoán số câu trả lời khác SFT tăng lên (hiện chỉ 20/58 khác), câu dài hơn, và có thể xuất hiện dịch
chuyển xác suất (chosen giảm). Với β = 0.5, reward β·log-ratio có thang lớn hơn nên margin danh nghĩa có thể lớn
hơn, nhưng policy gần như không rời reference: số câu giống hệt SFT tăng và win rate càng sát 0.5.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định: giữ cấu hình DPO mặc định (β = 0.1, lr = 5e-6, 1 epoch, 800 cặp) thay vì tăng cường độ huấn luyện.**

1. **Phương án thay thế.** Tôi có thể tăng tốc độ học lên 2e-5 hoặc chạy 2–3 epoch, hoặc giảm β xuống 0.05 để mô hình
   thay đổi mạnh hơn. Kết quả cho thấy cấu hình hiện tại quá "nhẹ tay": 38/58 câu trả lời giống hệt SFT.
2. **Vì sao chọn mặc định.** Thứ nhất là ngân sách: trên T4 miễn phí, riêng NB3 đã mất khoảng 44 phút và cả pipeline
   khoảng 1,5 giờ; nhân đôi số epoch có nguy cơ hết giờ GPU giữa chừng và mất toàn bộ file. Thứ hai, tôi muốn một mốc
   cơ sở sạch để so: lr nhỏ và β = 0.1 là vùng an toàn trong bài báo DPO, ít rủi ro dịch chuyển xác suất hay phá hỏng
   khả năng ngôn ngữ đã học ở SFT. Thứ ba, dữ liệu có thiên vị độ dài (65,9% chosen dài hơn), nên huấn luyện mạnh hơn
   dễ khuếch đại "hack độ dài" hơn là chất lượng.
3. **Kết quả.** Một phần xác nhận và một phần bất ngờ. Xác nhận: đường reward an toàn, held-out đi cùng train, không
   học thuộc, không dịch chuyển xác suất. Bất ngờ: tác động lên hành vi sinh văn bản nhỏ đến mức hai phần ba câu trả
   lời không đổi một ký tự, nên win rate 0.56 [0.48, 0.64] không phân biệt được với 0.5. Ngay cả trong 20 cặp có khác
   biệt, lợi thế của DPO gần như biến mất khi khớp độ dài (0.523). Tôi cũng không ngờ cả rejected lẫn chosen đều tăng
   reward; điều này cho thấy DPO ở đây chủ yếu kéo mô hình về phong cách của dữ liệu on-policy.
4. **Làm lại thì đổi gì.** (a) Sửa template SFT để bỏ khối `<think>` rỗng, loại hai token rác `<tool_call>`. (b) Chạy
   β-sweep {0.05, 0.1} với 2 epoch và lấy mẫu có nhiệt độ thay vì chỉ greedy, để đo tác động thật lên phân phối đầu ra.
   (c) Dùng biến thể có chuẩn hoá độ dài (LD-DPO hoặc DPO-norm) hoặc lọc bớt các cặp chosen dài hơn hẳn, để tách
   "tốt hơn" khỏi "dài hơn". (d) Thêm một giám khảo API khác họ để kiểm tra rò rỉ sở thích.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | — | — | — | — |
| GSM8K | — | — | — | — |
| Global-MMLU-vi | — | — | — | — |

Không làm phần bonus này.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.65 | +0.080 | 661 ký tự (NB4) | lần chạy chính NB3 |
| RPO | — | — | — | không chạy |
| DPO-norm | — | — | — | không chạy |
| LD-DPO | — | — | — | không chạy |
| ORPO | — | — | — | không chạy |

Không làm phần bonus này.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | không chạy |
| Sai số chuẩn ≈ √(p(1−p)/n) | không chạy |

Không làm phần bonus này.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Hai phần ba câu trả lời sau DPO giống hệt SFT từng ký tự. Margin held-out dương và độ chính xác reward 0.65 trông như
"DPO đã học", nhưng gần như không đổi được đầu ra greedy. Reward ngầm và hành vi sinh văn bản là hai thước đo khác nhau.
