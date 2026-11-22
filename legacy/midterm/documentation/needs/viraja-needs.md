# Viraja's 8 information needs

My 8 of the group's information needs for the 120-query evaluation. Each need is
written four ways, per the plan: **hindi** (Devanagari), **hinglish** (natural
Roman), **messy** (sloppy Roman, the kind Aksharantar's test data looks like) and
**english**. One relevance judgment per need covers all four forms.

Need ids are `V01`–`V08` here so they don't collide; renumber them into the
master `N##` sequence when we assemble the full 120.

| Need | What the user wants |
|---|---|
| V01 | Tomorrow's weather / rain alert for Delhi |
| V02 | India–Australia cricket match, Kohli's century |
| V03 | Bihar election dates announced |
| V04 | Stock market rally, Sensex up |
| V05 | School holiday because of heavy rain |
| V06 | Today's petrol and diesel prices |
| V07 | PM inaugurates a new railway line |
| V08 | Mumbai monsoon withdrawal, clear weather |

## Queries (ready to append to `queries.tsv`)
Tab-separated: `qid` · `need_id` · `form` · `query`.

```tsv
qid	need_id	form	query
V01_hi	V01	hindi	कल दिल्ली में मौसम और बारिश का अलर्ट
V01_hinglish	V01	hinglish	kal delhi mein mausam aur barish ka alert
V01_messy	V01	messy	kal dilli me mosam aur baarish ka alrt
V01_en	V01	english	delhi weather rain alert tomorrow
V02_hi	V02	hindi	भारत ऑस्ट्रेलिया मैच में कोहली का शतक
V02_hinglish	V02	hinglish	bharat australia match mein kohli ka shatak
V02_messy	V02	messy	india australia match me kohli century
V02_en	V02	english	india australia cricket kohli century
V03_hi	V03	hindi	बिहार चुनाव की तारीखें घोषित
V03_hinglish	V03	hinglish	bihar chunav ki tareekhein ghoshit
V03_messy	V03	messy	bihar election ki date announce
V03_en	V03	english	bihar assembly election dates
V04_hi	V04	hindi	शेयर बाजार में तेजी सेंसेक्स चढ़ा
V04_hinglish	V04	hinglish	share bazaar mein teji sensex chadha
V04_messy	V04	messy	share market tezi sensex up
V04_en	V04	english	stock market sensex rally today
V05_hi	V05	hindi	भारी बारिश के कारण स्कूलों में छुट्टी
V05_hinglish	V05	hinglish	bhari barish ke karan schoolon mein chutti
V05_messy	V05	messy	rain ki wajah se school holiday
V05_en	V05	english	school holiday due to heavy rain
V06_hi	V06	hindi	आज पेट्रोल डीजल के दाम
V06_hinglish	V06	hinglish	aaj petrol diesel ke daam
V06_messy	V06	messy	petrol diesel price aaj
V06_en	V06	english	petrol diesel price today
V07_hi	V07	hindi	प्रधानमंत्री नई रेल लाइन का उद्घाटन
V07_hinglish	V07	hinglish	pradhanmantri nayi rail line ka udghatan
V07_messy	V07	messy	pm new rail line inauguration
V07_en	V07	english	prime minister new railway line inauguration
V08_hi	V08	hindi	मुंबई में मानसून की विदाई मौसम साफ
V08_hinglish	V08	hinglish	mumbai mein monsoon ki vidai mausam saaf
V08_messy	V08	messy	mumbai monsoon vapsi mosam saf
V08_en	V08	english	mumbai monsoon withdrawal clear weather
```

The **messy** form is the one that exercises my layer hardest: `mosam`, `dilli`,
`baarish`, `alrt`, `saf` all need phonetic matching to reach the Devanagari
articles. These needs line up with the corpus sections (weather, sports,
politics, business, education, national).
