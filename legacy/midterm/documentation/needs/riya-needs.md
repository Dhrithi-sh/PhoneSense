# Riya's 8 information needs

My 8 of the group's information needs for the 120-query evaluation. Each need is
written four ways, per the plan: **hindi** (Devanagari), **hinglish** (natural
Roman), **messy** (sloppy Roman with phonetic variants) and **english**. One
relevance judgment per need covers all four forms.

Need ids are `Y01`–`Y08` so they don't collide with Rishit (`R01`–`R08`), Viraja
(`V01`–`V08`), or Dhrithi (`D01`–`D08`), matching the `PREFIX["riya"] = "Y"`
mapping in `app/pages/judge.py`.

Every need here was selected from prominent news stories present in the crawled
corpus across multiple regional publications (*Dainik Jagran*, *Amar Ujala*,
*Navbharat Times*, *Live Hindustan*, *Aaj Tak*). For my part of the project (P1
— Regional News Crawler, Corpus, Deduplication & Shared Tooling), these needs
specifically test:

- **Near-duplicate wire stories & reprints:** Syndicated national stories (gold
  prices, DA hike, Supreme Court orders, festival specials) reported across
  multiple papers to test duplicate clustering and source authority.
- **Regional Hindi-belt geographic coverage:** Weather and administrative
  governance stories spanning UP, Delhi, and Bihar to test city/state metadata
  filtering.
- **Phonetic and cross-lingual matching:** Noisy Roman variations (`baarish`,
  `teohar`, `buldozer`, `mehangai`) to exercise Soundex/Dhvani-code matching,
  along with English queries to exercise dictionary translation.

| Need | What the user wants | Rough number of matching articles in corpus |
|---|---|---|
| Y01 | Heavy rain alert and temperature drop in Uttar Pradesh and Lucknow | 8+ |
| Y02 | Gold and silver prices hit record highs ahead of Diwali and Dhanteras | 7+ |
| Y03 | Indian Railways announces festival special trains for Diwali and Chhath Puja | 9+ |
| Y04 | Supreme Court hearing and guidelines on bulldozer action and demolitions | 6+ |
| Y05 | India vs Bangladesh T20 series team performance and match results | 8+ |
| Y06 | Israel airstrikes in Lebanon and Middle East conflict escalation | 5+ |
| Y07 | UP Chief Minister Yogi Adityanath law and order and development review | 6+ |
| Y08 | Central Government Dearness Allowance (DA) hike for employees | 6+ |

## Queries (ready to append to `queries.tsv`)
Tab-separated: `qid` · `need_id` · `form` · `query`.

```tsv
qid	need_id	form	query
Y01_hi	Y01	hindi	उत्तर प्रदेश लखनऊ में भारी बारिश और तापमान अलर्ट
Y01_hinglish	Y01	hinglish	uttar pradesh lucknow mein bhari barish aur tapman alert
Y01_messy	Y01	messy	up lucknow me bhaari baarish or tapman girawat
Y01_en	Y01	english	heavy rain and temperature alert in uttar pradesh lucknow
Y02_hi	Y02	hindi	दीपावली धनतेरस से पहले सोना और चांदी के दाम रिकॉर्ड स्तर पर
Y02_hinglish	Y02	hinglish	deepawali dhanteras se pehle sona aur chandi ke daam record par
Y02_messy	Y02	messy	diwali dhanteras pe gold silver sona chandi bhav teji
Y02_en	Y02	english	gold and silver prices record high ahead of diwali dhanteras
Y03_hi	Y03	hindi	दिवाली और छठ पूजा पर रेलवे की स्पेशल ट्रेनें
Y03_hinglish	Y03	hinglish	diwali aur chhath puja par railway ki special trainein
Y03_messy	Y03	messy	diwali chath puja k liye railway festival special train list
Y03_en	Y03	english	indian railways special trains for diwali and chhath puja
Y04_hi	Y04	hindi	सुप्रीम कोर्ट में बुलडोजर कार्रवाई पर सुनवाई और दिशा निर्देश
Y04_hinglish	Y04	hinglish	supreme court mein bulldozer karrawai par sunwai aur disha nirdesh
Y04_messy	Y04	messy	sc me buldozer action sunvai guidelines
Y04_en	Y04	english	supreme court hearing and guidelines on bulldozer demolition action
Y05_hi	Y05	hindi	भारत बांग्लादेश टी20 मैच में भारतीय टीम का प्रदर्शन
Y05_hinglish	Y05	hinglish	bharat bangladesh t20 match mein bhartiya team ka pradarshan
Y05_messy	Y05	messy	india bangladesh t20 match me team india performance
Y05_en	Y05	english	india bangladesh t20 cricket match team performance
Y06_hi	Y06	hindi	लेबनान में इजरायल के हवाई हमले और तनाव
Y06_hinglish	Y06	hinglish	lebanon mein israel ke hawai hamle aur tanav
Y06_messy	Y06	messy	lebanan me israel air strike attack or middle east war
Y06_en	Y06	english	israel air strikes in lebanon middle east conflict
Y07_hi	Y07	hindi	मुख्यमंत्री योगी आदित्यनाथ की कानून व्यवस्था और विकास समीक्षा बैठक
Y07_hinglish	Y07	hinglish	mukhyamantri yogi adityanath ki kanoon vyavastha aur vikas samiksha baithak
Y07_messy	Y07	messy	cm yogi adityanath up law and order crime meeting review
Y07_en	Y07	english	cm yogi adityanath law and order development review meeting up
Y08_hi	Y08	hindi	केंद्रीय कर्मचारियों के महंगाई भत्ते में तीन प्रतिशत की बढ़ोतरी
Y08_hinglish	Y08	hinglish	kendriya karmachariyon ke mehangai bhatte mein teen pratishat ki badhotari
Y08_messy	Y08	messy	central govt employee da hike mehangai bhatta 3 percent badha
Y08_en	Y08	english	central government employees dearness allowance da hike announcement
```
