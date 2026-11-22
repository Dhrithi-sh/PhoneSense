# Failure cases (test split)

System: `bm25_tuned` (best on dev nDCG@10). Reference: `bm25_tuned`.

## q34 (Lowest nDCG@10): Motorola Edge

nDCG@10 = 0.233, bm25_tuned nDCG@10 = 0.233. Filters (this run applies no filter): `none`

| Rank | doc_id | Title | Key specs | Grade |
|---:|---|---|---|---|
| 1 | B00658HKTK | Motorola RAZR V3xx Platinum No Contract AT&T Cell Phone | 0.5GB | 0 |
| 2 | B004FLL6SU | Motorola EX112 Unlocked Phone with Full QWERTY Keyboard, 3.15 MP Camer | 32GB | 1 |
| 3 | B0BLR46Y13 | Motorola Edge + /2022/ 4800mAh Battery / Unlocked / Made for US 8/512G | $499.99, 8GB RAM, 512GB, 4800mAh, 6.7", 5G, android | 1 |
| 4 | B00O2SW2IY | Verizon 4G LTE Moto Luge by Motorola Prepaid Smartphone | 1GB RAM, 32GB, 4.3", android | 0 |
| 5 | B07MX8DHBJ | Motorola MOT19227BVB G6 Play 16GB Prepaid Virgin Mobile Smartphone, Bl | $79.44, 16GB, 5.7", android | 1 |

Missed relevant docs (grade >= 2, not in the top 10):

- B09FZSR9N5 (grade 3, rank 19): Motorola Edge | 2021 | 2-Day Battery | Unlocked | Made for US by Motor

Why: 

## q08 (Lowest nDCG@10): cheap phone under $150 with 64GB storage

nDCG@10 = 0.234, bm25_tuned nDCG@10 = 0.234. Filters (this run applies no filter): `none`

| Rank | doc_id | Title | Key specs | Grade |
|---:|---|---|---|---|
| 1 | B08FT1NWGR | Xgody 6.8 Inch Unlocked Smartphones, K30 Android 10.0 Cell Phone Cheap | 1GB RAM, 8GB, 3000mAh, 6.8", android | 1 |
| 2 | B09NY9L6TS | Unlocked Phone KXD 6C Android Cell Phones 5.5" Full Screen 4G Dual SIM | 0.5GB RAM, 16GB, 2500mAh, 5.5", android | 1 |
| 3 | B09BKX9YMG | KXD 6A Unlocked Cell Phone / 3G Smartphone / 5.5” Full-Screen Display  | $49.99, 1GB RAM, 8GB, 2500mAh, 5.5", android | 1 |
| 4 | B0B3DR43GC | Xgody X60 Pro Unlocked Smartphones, 6.5 Inch Android 9.1 OS Cheap Cell | $74.99, 8GB, 3000mAh, 6.5", android | 1 |
| 5 | B09FSC5BXY | CUBOT Max 3P Unlocked Smartphone, Android 11 Cell Phone，6.95" Screen，4 | $79.99, 4GB RAM, 64GB, 5000mAh, 6.95", android | 3 |

Missed relevant docs (grade >= 2, not in the top 10):

- B0BBV63XNQ (grade 3, rank 18): Ulefone Unlocked Smartphones, Note 6T Latest Android 12 3GB+64GB Cell 
- B0C1Q3SLTT (grade 3, rank 76): SAMSUNG Galaxy A14 5G A Series Cell Phone, Factory Unlocked Android Sm
- B0C9JB67GX (grade 3, not retrieved): Simple Mobile Motorola Moto g Power, 64GB, Black - Prepaid Smartphone 
- B0074RGE8E (grade 2, rank 69): Apple iPhone 4S 64GB (White) - Verizon
- B017BAJTXU (grade 2, not retrieved): Huawei Mate S 64GB Gold
- B01G7PGCN0 (grade 2, not retrieved): Apple iPhone SE 64 GB Verizon, Gold
- B0714DP3BG (grade 2, not retrieved): Apple iPhone 8 Plus. 64gb Unlocked (Gold)
- B07CQ7FXVY (grade 2, not retrieved): Apple iPhone 8 64GB Unlocked GSM 4G LTE Phone - Red
- B07JZGGLRD (grade 2, not retrieved): BLU VIVO 8-5.5" Full HD, 4G LTE International Version Smartphone -64GB
- B07WPX7VCG (grade 2, rank 21): Unlocked Cell Phone, OUKITEL C17 Pro 64GB+4GB Unlocked phones 13+5MP T

Why: 

## q42 (Lowest nDCG@10): phone that does everything well

nDCG@10 = 0.334, bm25_tuned nDCG@10 = 0.334. Filters (this run applies no filter): `none`

| Rank | doc_id | Title | Key specs | Grade |
|---:|---|---|---|---|
| 1 | B00D41P12O | HTC ONE (M7) 64gb of Onboard Memory / Unlocked Bootloader. | 64GB, android | 1 |
| 2 | B00J00HMU2 | Unlocked ZTE Z222 Flip Phone With Camera For ATT, T-Mobile and Other S | 2GB | 1 |
| 3 | B004LRVNG8 | Dell Venue Pro Phone 8 GB with Windows 7 Operating System, 5MP Camera  | 8GB, other | 2 |
| 4 | B00KVX4EL2 | Huawei Ascend Y530, 4GB, Factory Unlocked GSM Android Smartphone - Bla | 0.5GB RAM, 4GB, 4.5", android | 1 |
| 5 | B008D09OR4 | Sony Xperia P LT22i-BK Unlocked Phone with 8 MP Camera, Android 2.3 OS | 1GB RAM, 16GB, 4", android | 1 |

Missed relevant docs (grade >= 2, not in the top 10):

- B0BHF3RYQQ (grade 3, rank 61): Simple Mobile Samsung Galaxy A53 5G, 128GB, Black - Prepaid Smartphone

Why: 
