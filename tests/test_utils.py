from pathlib import Path

from smer_visual import utils


class FakeToken:
    def __init__(self, lemma):
        self.lemma_ = lemma


def test_get_image_files_with_class_filters_extensions_and_checkpoints(tmp_path):
    cat_dir = tmp_path / "cats"
    dog_dir = tmp_path / "dogs"
    checkpoint_dir = tmp_path / ".ipynb_checkpoints"
    cat_dir.mkdir()
    dog_dir.mkdir()
    checkpoint_dir.mkdir()

    (cat_dir / "first.jpg").write_bytes(b"cat")
    (dog_dir / "second.PNG").write_bytes(b"dog")
    (dog_dir / "notes.txt").write_text("ignore")
    (checkpoint_dir / "skip.png").write_bytes(b"skip")

    results = list(utils._get_image_files_with_class(str(tmp_path)))

    assert set(results) == {
        (str(cat_dir / "first.jpg"), "cats"),
        (str(dog_dir / "second.PNG"), "dogs"),
    }


def test_encode_image_returns_base64_string(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"binary-data")

    encoded = utils._encode_image(str(image_path))

    assert encoded == "YmluYXJ5LWRhdGE="


def test_preprocess_text_lemmatizes_and_strips_punctuation(monkeypatch):
    lemma_map = {
        "Cats": ["cat"],
        "running": ["run"],
        "brown foxes": ["brown", "fox"],
    }

    def fake_load(_name):
        def fake_nlp(text):
            return [FakeToken(lemma) for lemma in lemma_map[text]]

        return fake_nlp

    monkeypatch.setattr(utils.spacy, "load", fake_load)

    result = utils._preprocess_text("Cats!, running, , brown foxes.")

    assert result == "cat,run,brown fox"


def test_preprocess_text_downloads_model_when_missing(monkeypatch):
    load_calls = []
    download_calls = []

    def fake_load(_name):
        load_calls.append(True)
        if len(load_calls) == 1:
            raise OSError("missing model")
        return lambda text: [FakeToken(text.upper())]

    monkeypatch.setattr(utils.spacy, "load", fake_load)
    monkeypatch.setattr(utils.spacy.cli, "download", lambda name: download_calls.append(name))

    result = utils._preprocess_text("word")

    assert result == "WORD"
    assert download_calls == ["en_core_web_sm"]
    assert len(load_calls) == 2
