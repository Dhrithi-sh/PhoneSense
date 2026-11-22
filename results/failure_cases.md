# Failure cases (test split)

System: `rrf_k5_bm25_tuned+dense_sentence-transformers-all-minilm-l6-v2` (best on dev nDCG@10). Reference: `bm25_tuned`.

## q08 (Lowest nDCG@10): cheap phone under $150 with 64GB storage

nDCG@10 = 0.321, bm25_tuned nDCG@10 = 0.234. Filters (this run applies no filter): `none`

| Rank | doc_id | Title | Key specs | Grade |
|---:|---|---|---|---|
| 1 | B002TX7ABI | Nokia 6350 Phone, Grey (AT&T) |  | 1 |
| 2 | B08FT1NWGR | Xgody 6.8 Inch Unlocked Smartphones, K30 Android 10.0 Cell Phone Cheap | 1GB RAM, 8GB, 3000mAh, 6.8", android | 1 |
| 3 | B07V26FP2S | Mini Smartphone Android Phone 2.5" MTK6580A Quad Core 1GB 8GB Android  | 1GB RAM, 8GB, 1050mAh, android | 0 |
| 4 | B09NY9L6TS | Unlocked Phone KXD 6C Android Cell Phones 5.5" Full Screen 4G Dual SIM | 0.5GB RAM, 16GB, 2500mAh, 5.5", android | 1 |
| 5 | B002TX79HS | Nokia 6350, Red (AT&T) | 0.0508GB | 1 |

Missed relevant docs (grade >= 2, not in the top 10):

- B0BBV63XNQ (grade 3, rank 36): Ulefone Unlocked Smartphones, Note 6T Latest Android 12 3GB+64GB Cell 
- B0C1Q3SLTT (grade 3, not retrieved): SAMSUNG Galaxy A14 5G A Series Cell Phone, Factory Unlocked Android Sm
- B0074RGE8E (grade 2, rank 73): Apple iPhone 4S 64GB (White) - Verizon
- B017BAJTXU (grade 2, rank 33): Huawei Mate S 64GB Gold
- B01G7PGCN0 (grade 2, not retrieved): Apple iPhone SE 64 GB Verizon, Gold
- B0714DP3BG (grade 2, not retrieved): Apple iPhone 8 Plus. 64gb Unlocked (Gold)
- B07CQ7FXVY (grade 2, not retrieved): Apple iPhone 8 64GB Unlocked GSM 4G LTE Phone - Red
- B07JZGGLRD (grade 2, not retrieved): BLU VIVO 8-5.5" Full HD, 4G LTE International Version Smartphone -64GB
- B07WPX7VCG (grade 2, rank 42): Unlocked Cell Phone, OUKITEL C17 Pro 64GB+4GB Unlocked phones 13+5MP T
- B07WZJMJD8 (grade 2, rank 43): Unlocked Cell Phone, OUKITEL C17 Pro 64GB+4GB Unlocked phones 13+5MP T

Why: 

## q10 (Lowest nDCG@10): 2023 phone with 5G support

nDCG@10 = 0.350, bm25_tuned nDCG@10 = 0.883. Filters (this run applies no filter): `none`

| Rank | doc_id | Title | Key specs | Grade |
|---:|---|---|---|---|
| 1 | B09XG1QW1Y | total wireless Blu View 2 (2022) 4G LTE Prepaid Smartphone (Locked) -  | 13GB RAM, 32GB, 2920mAh, 5.45", android | 0 |
| 2 | B0C1Q3SLTT | SAMSUNG Galaxy A14 5G A Series Cell Phone, Factory Unlocked Android Sm | $147.5, 4GB RAM, 64GB, 5000mAh, 6.6", 5G, android | 3 |
| 3 | B01AH1BWOO | BLU VIVO 5 Smartphone -5.5" 4G LTE GSM Unlocked - 32GB +3GB RAM - Silv | 32GB, 5.5", android | 0 |
| 4 | B0CCNVVTZ9 | DOOGEE V MAX (2023) 5G Rugged Smartphone, 22000mAh 20GB+256GB Android  | $463.83, 12GB RAM, 20GB, 6.58", 5G, android | 1 |
| 5 | B09HLWX1JW | Nokia G50 5G / Android 11 / Unlocked Smartphone / US Version / 4/128GB | $194.95, 4GB RAM, 128GB, 5000mAh, 6.82", 5G, android | 0 |

Missed relevant docs (grade >= 2, not in the top 10):

- B0BS3XJTBP (grade 3, rank 17): Samsung Galaxy S23 Ultra Cell Phone + $100 Amazon Gift Card Bundle, Fa
- B0BS3XWZMX (grade 3, rank 14): Samsung Galaxy S23 Ultra Cell Phone + $100 Amazon Gift Card Bundle, Fa
- B0BSP2TF24 (grade 3, rank 12): BLU F92e 5G | 2023 | 3-Day Battery | Unlocked | 6.5” HD+ Display | 128

Why: 

## q42 (Lowest nDCG@10): phone that does everything well

nDCG@10 = 0.445, bm25_tuned nDCG@10 = 0.334. Filters (this run applies no filter): `none`

| Rank | doc_id | Title | Key specs | Grade |
|---:|---|---|---|---|
| 1 | B00D41P12O | HTC ONE (M7) 64gb of Onboard Memory / Unlocked Bootloader. | 64GB, android | 1 |
| 2 | B07J66VJDC | Phones | 64GB, 10000mAh, 5.7", android | 1 |
| 3 | B00009LI9X | Nokia 3560 Phone (AT&T) |  | 1 |
| 4 | B00J00HMU2 | Unlocked ZTE Z222 Flip Phone With Camera For ATT, T-Mobile and Other S | 2GB | 1 |
| 5 | B003D7HW7G | Samsung SGH-a237 Quad-band Cell Phone | 0.0078GB, other | 1 |

Missed relevant docs (grade >= 2, not in the top 10):

- B0BHF3RYQQ (grade 3, not retrieved): Simple Mobile Samsung Galaxy A53 5G, 128GB, Black - Prepaid Smartphone

Why: 

## q43 (Largest loss vs. bm25_tuned): phone for content creators

nDCG@10 = 0.477, bm25_tuned nDCG@10 = 0.817. Filters (this run applies no filter): `none`

| Rank | doc_id | Title | Key specs | Grade |
|---:|---|---|---|---|
| 1 | B0033SFC4U | Lg Ct810 Incite Unlocked GSM Cell Phone - 3.2 Megapixel Camera, Video  |  | 0 |
| 2 | B09W3PK64J | Unlocked S22 Plus 5G - 128GB - Green - SM-S906UZGAXAA (Renewed) | $400, 8GB, 5G, android | 2 |
| 3 | B000E8FS2I | Nokia 6030 (AT&T) |  | 0 |
| 4 | B0B4VCQ7MB | SAMSUNG Galaxy S22+ Plus 256GB Factory Unlocked SM-S906U1 Pink Gold (R | $493, 256GB, 5G, android | 2 |
| 5 | B00009LI9X | Nokia 3560 Phone (AT&T) |  | 0 |

Missed relevant docs (grade >= 2, not in the top 10):

- B079XYJ4FJ (grade 3, rank 12): LG V20 64GB H918 - Unlocked by T-Mobile for all GSM Carriers (Titan Gr

Why: 

## q09 (Largest loss vs. bm25_tuned): flagship phone with 12GB RAM and 512GB storage

nDCG@10 = 0.708, bm25_tuned nDCG@10 = 0.936. Filters (this run applies no filter): `none`

| Rank | doc_id | Title | Key specs | Grade |
|---:|---|---|---|---|
| 1 | B07XPH822S | ASUS ROG Phone 2 (New) Unlocked GSM US Version & Warranty, 512GB Stora | 12GB RAM, 512GB, 6.6", android | 3 |
| 2 | B09NQZJ4VY | Ulefone 7700mAh Battery Android Smartphone, Note 12P Android 11 Octa-c | $99.99, 4GB RAM, 64GB, 7700mAh, 6.82", android | 0 |
| 3 | B081VVW3BJ | ASUS ROG Gaming Phone II (ZS660KL-S855P-12G512G-BK) 6.6" FHD+ 2340x108 | 12GB RAM, 512GB, 6.6" | 3 |
| 4 | B0BMPVZ1BS | 12Pro MAX Unlocked Cellphone, 6.26in HD Screen Mobile Phone, 1+8G Dual | $126.66, 1GB RAM, 8GB, 1950mAh, 6.3", 5G, android | 0 |
| 5 | B0825DKC2F | ASUS ROG Phone 2 (New) Unlocked GSM US Version & Warranty, 1TB Storage | 12GB RAM, 1024GB, 6.6" | 3 |

Missed relevant docs (grade >= 2, not in the top 10):

- B09TRZGKXW (grade 3, rank 12): Samsung Galaxy S22 Ultra S9080 5G 512GB 12GB RAM Factory Unlocked (GSM

Why: 
