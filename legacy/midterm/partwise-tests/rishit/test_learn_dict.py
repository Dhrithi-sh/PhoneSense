from dhvani.rank.learn_dict import headline_pairs, learn, split_headline, write_dict
from dhvani.rank.xling import Translator, load_news_dict


def test_split_finds_the_english_tail():
    assert split_headline("भूकंप से दहशत - earthquake tremors felt") == ("भूकंप से दहशत", "earthquake tremors felt")
    assert split_headline("दिल्ली में बारिश") is None


def test_pairs_drop_stop_words_and_short_words():
    pairs = headline_pairs([{"headline": "भूकंप से दहशत - the earthquake news in delhi"}])
    assert pairs == [({"भूकंप", "से", "दहशत"}, {"earthquake", "delhi"})]


def _arts(*rows):
    return [{"headline": f"{h} - {e}"} for h, e in rows]


def test_words_that_keep_appearing_together_are_aligned():
    rows = [("भूकंप दहशत", "earthquake panic"), ("भूकंप असम", "earthquake assam"), ("भूकंप उत्तराखंड", "earthquake uttarakhand"),
            ("बारिश मुंबई", "rain mumbai"), ("बारिश पटना", "rain patna"), ("बारिश जयपुर", "rain jaipur")] + \
           [(f"खबर{i}", f"filler{i}") for i in range(60)]
    learned = learn(headline_pairs(_arts(*rows)), min_pairs=3, min_dice=0.5)
    assert learned["earthquake"][0] == "भूकंप" and learned["rain"][0] == "बारिश"
    assert "panic" not in learned          # seen once, below min_pairs


def test_written_file_loads_with_the_news_dict_reader(tmp_path):
    path = write_dict({"earthquake": ("भूकंप", 0.9, 5)}, tmp_path / "learned.tsv")
    assert load_news_dict(path) == {"earthquake": [("भूकंप", 1.0)]}


def test_hand_dictionary_wins_over_learned(tmp_path):
    learned = tmp_path / "learned.tsv"
    learned.write_text("rain\tबरसात\nzzzword\tभूकंप\n", encoding="utf-8")
    t = Translator(muse_path=tmp_path / "none.txt", learned_path=learned)
    assert dict(t.lookup("rain")) == {"बारिश": 0.7, "वर्षा": 0.3}     # hand-made entry wins
    assert dict(t.lookup("zzzword")) == {"भूकंप": 1.0}                # learned fills the gap
