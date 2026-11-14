# Rishit's 8 information needs

My 8 of the group's information needs for the 120-query evaluation. Each need is
written four ways, same as the others: **hindi** (Devanagari), **hinglish**
(natural Roman), **messy** (sloppy Roman) and **english**. One relevance
judgment per need covers all four forms.

Need ids are `R01`–`R08` so they don't collide; renumber them into the master
`N##` sequence when we assemble the full 120.

Every need here was picked from stories that are actually in Riya's first
crawl (6 to 7 October 2026), and each one has at least two matching articles,
usually from different papers. That matters for my parts of the project:

- several needs are the same story carried by many papers (earthquake, Char
  Dham, Shreyas Iyer), so they test duplicate collapsing and first to publish
- the english forms use names and words our dictionary has to translate
  (earthquake, election, captain, century), so they test the cross-lingual step
- the protest and Rajya Sabha needs have a lot of near-miss articles around
  them, so precision actually gets tested and not just recall

| Need | What the user wants | Rough number of matching articles in the 300 |
|---|---|---|
| R01 | Earthquake felt in Delhi-NCR, centre in Uttarakhand | 8 |
| R02 | Shreyas Iyer's first T20I century against West Indies | 6 |
| R03 | Rahul Gandhi detained during the opposition march against the CEC | 10+ |
| R04 | Char Dham yatra breaks the 2023 record | 3 |
| R05 | UP Rajya Sabha candidates, their nominations and assets | 4 |
| R06 | Bypoll voting in Nandigram, West Bengal and Tamil Nadu | 3 |
| R07 | Smriti Mandhana named India women's captain | 3 |
| R08 | Trump links the France protests to Islam | 3 |

The counts are from reading the headlines only; the real relevance judgments
come from pooling once the full crawl is indexed.

## Queries (ready to append to `queries.tsv`)
Tab-separated: `qid` · `need_id` · `form` · `query`.

```tsv
qid	need_id	form	query
R01_hi	R01	hindi	दिल्ली एनसीआर में भूकंप के झटके उत्तराखंड
R01_hinglish	R01	hinglish	delhi ncr mein bhukamp ke jhatke uttarakhand
R01_messy	R01	messy	dilli ncr bhukamp jhatke uttrakhand
R01_en	R01	english	earthquake tremors delhi ncr uttarakhand
R02_hi	R02	hindi	श्रेयस अय्यर का टी20 शतक वेस्टइंडीज
R02_hinglish	R02	hinglish	shreyas iyer ka t20 shatak west indies
R02_messy	R02	messy	shreyas ayyar t20 century wi
R02_en	R02	english	shreyas iyer t20 century west indies
R03_hi	R03	hindi	राहुल गांधी हिरासत चुनाव आयोग के खिलाफ मार्च
R03_hinglish	R03	hinglish	rahul gandhi hirasat chunav aayog ke khilaf march
R03_messy	R03	messy	rahul gandhi detain cec march protest
R03_en	R03	english	rahul gandhi detained protest against election commission
R04_hi	R04	hindi	चारधाम यात्रा ने तोड़ा रिकॉर्ड श्रद्धालु
R04_hinglish	R04	hinglish	chardham yatra ne toda record shraddhalu
R04_messy	R04	messy	char dham yatra record tuta
R04_en	R04	english	char dham yatra record pilgrims
R05_hi	R05	hindi	यूपी राज्यसभा प्रत्याशी नामांकन संपत्ति
R05_hinglish	R05	hinglish	up rajya sabha pratyashi namankan sampatti
R05_messy	R05	messy	up rajyasabha candidate nomination property
R05_en	R05	english	uttar pradesh rajya sabha candidates nomination assets
R06_hi	R06	hindi	नंदीग्राम उपचुनाव में वोटिंग
R06_hinglish	R06	hinglish	nandigram upchunav mein voting
R06_messy	R06	messy	nandigram by election voting kitni hui
R06_en	R06	english	nandigram bypoll voting turnout
R07_hi	R07	hindi	स्मृति मंधाना भारतीय महिला टीम की कप्तान
R07_hinglish	R07	hinglish	smriti mandhana bhartiya mahila team ki kaptan
R07_messy	R07	messy	smriti mandana women team captain bani
R07_en	R07	english	smriti mandhana india women captain
R08_hi	R08	hindi	फ्रांस प्रदर्शन पर ट्रंप का इस्लाम बयान
R08_hinglish	R08	hinglish	france pradarshan par trump ka islam bayan
R08_messy	R08	messy	trump france protest islam statement
R08_en	R08	english	trump france protests islam claim
```
