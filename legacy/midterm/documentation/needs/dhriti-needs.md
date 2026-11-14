# Dhriti's 8 information needs

My 8 of the group's information needs for the 120-query evaluation. Each need is
written four ways, same as the others: **hindi** (Devanagari), **hinglish**
(natural Roman), **messy** (sloppy Roman) and **english**. One relevance
judgment per need covers all four forms.

Need ids are `D01`–`D08` so they don't collide; renumber them into the master
`N##` sequence when we assemble the full 120.

Every need is a real story in Riya's crawl (6–7 October 2026) with at least two
matching articles. They lean on words with heavy inflection and spelling
variation (मुठभेड़/मुठभेड़ों, प्रदूषण, पुरस्कार, हत्याकांड), which is exactly
what the normalizer and stemmer have to fold together.

| Need | What the user wants |
|---|---|
| D01 | PM Modi completes 25 years in public life |
| D02 | Physics Nobel for the neutrino discovery (Francis Halzen) |
| D03 | Delhi-NCR air-pollution crackdown and curbs |
| D04 | Patna Metro completes one year |
| D05 | Sitamarhi police encounter in the Dinesh Yadav murder case |
| D06 | GST Council meeting on 8 October and its announcements |
| D07 | Four cricketers killed in a UP road accident |
| D08 | Nissan's Navratri festive discount on its SUVs |

## Queries (ready to append to `queries.tsv`)
Tab-separated: `qid` · `need_id` · `form` · `query`.

```tsv
qid	need_id	form	query
D01_hi	D01	hindi	प्रधानमंत्री मोदी के जनसेवा के 25 साल पूरे
D01_hinglish	D01	hinglish	pradhanmantri modi ke janseva ke 25 saal poore
D01_messy	D01	messy	pm modi 25 saal public life
D01_en	D01	english	pm modi completes 25 years in public life
D02_hi	D02	hindi	न्यूट्रिनो की खोज के लिए फिजिक्स नोबेल पुरस्कार
D02_hinglish	D02	hinglish	neutrino ki khoj ke liye physics nobel puraskar
D02_messy	D02	messy	neutrino physics nobel prize halzen
D02_en	D02	english	physics nobel prize neutrino discovery
D03_hi	D03	hindi	दिल्ली एनसीआर में प्रदूषण पर शिकंजा
D03_hinglish	D03	hinglish	delhi ncr mein pradushan par shikanja
D03_messy	D03	messy	delhi ncr pollution par rok
D03_en	D03	english	delhi ncr air pollution crackdown
D04_hi	D04	hindi	पटना मेट्रो का एक साल पूरा
D04_hinglish	D04	hinglish	patna metro ka ek saal poora
D04_messy	D04	messy	patna metro 1 saal complete
D04_en	D04	english	patna metro completes one year
D05_hi	D05	hindi	सीतामढ़ी में पुलिस मुठभेड़ दिनेश यादव हत्याकांड
D05_hinglish	D05	hinglish	sitamarhi mein police muthbhed dinesh yadav hatyakand
D05_messy	D05	messy	sitamarhi encounter dinesh yadav murder
D05_en	D05	english	sitamarhi police encounter dinesh yadav murder case
D06_hi	D06	hindi	जीएसटी काउंसिल की बैठक 8 अक्टूबर बड़े ऐलान
D06_hinglish	D06	hinglish	gst council ki baithak 8 october bade ailan
D06_messy	D06	messy	gst council meeting 8 oct
D06_en	D06	english	gst council meeting october 8 announcements
D07_hi	D07	hindi	यूपी में सड़क हादसे में चार क्रिकेटरों की मौत
D07_hinglish	D07	hinglish	up mein sadak hadse mein chaar cricketeron ki maut
D07_messy	D07	messy	up road accident 4 cricketers died
D07_en	D07	english	four cricketers killed up road accident
D08_hi	D08	hindi	नवरात्रि पर निसान की एसयूवी पर भारी छूट
D08_hinglish	D08	hinglish	navratri par nissan ki suv par bhari chhoot
D08_messy	D08	messy	navratri nissan suv discount offer
D08_en	D08	english	nissan navratri suv festive discount
```
