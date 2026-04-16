from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from smer_visual import smer


class FakeTensor:
    def __init__(self, value):
        self.value = np.array(value)

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return FakeTensor(self.value[item])


class FakeInputs(dict):
    def __init__(self):
        super().__init__({"input_ids": FakeTensor([[1, 2, 3]])})
        self.input_ids = "input-ids"

    def to(self, _device):
        return self


class FakeLocalEmbeddingModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **_inputs):
        return SimpleNamespace(last_hidden_state=FakeTensor([[[1.0, 2.0, 3.0]]]))


class FakeLocalCaptionModel:
    def __init__(self):
        self.device = "cpu"

    def generate(self, **_inputs):
        return np.array([[1, 2, 3]])


class FakeTokenizer:
    def __call__(self, word, **_kwargs):
        if word == "bad":
            raise RuntimeError("bad token")
        return {"input_ids": FakeTensor([[1, 2]])}

    def decode(self, _tokens, **_kwargs):
        return "assistant\n\nlocal caption"


class FakeProcessor:
    def apply_chat_template(self, _message, add_generation_prompt=True):
        assert add_generation_prompt is True
        return "prompt"

    def __call__(self, *_args, **_kwargs):
        return FakeInputs()

    def post_process_grounded_object_detection(
        self,
        _outputs,
        _input_ids,
        box_threshold,
        text_threshold,
        target_sizes,
    ):
        assert box_threshold == 0.4
        assert text_threshold == 0.3
        assert target_sizes == [(20, 10)]
        return [
            {
                "boxes": [np.array([1, 2, 5, 6])],
                "scores": [0.95],
                "labels": ["cat"],
            }
        ]


class FakePredictProbaModel:
    classes_ = np.array(["cat", "dog"])

    def predict_proba(self, embeddings):
        rows = []
        for emb in embeddings:
            score = float(np.sum(emb))
            p_cat = max(0.0, min(1.0, score / 10.0))
            rows.append([p_cat, 1.0 - p_cat])
        return np.array(rows)


class FakeBinaryModel:
    coef_ = np.array([[2.0, -1.0]])
    intercept_ = np.array([0.0])


class FakeMulticlassModel:
    coef_ = np.array([[1.0, 0.0], [0.0, 0.0]])
    intercept_ = np.array([0.0, 0.0])


class FakeImage:
    def __init__(self, size=(10, 20)):
        self.size = size
        self.saved_paths = []

    def copy(self):
        return self

    def save(self, path):
        self.saved_paths.append(str(path))


class FakeDraw:
    def __init__(self):
        self.rectangles = []
        self.texts = []

    def rectangle(self, coords, **kwargs):
        self.rectangles.append((coords, kwargs))

    def textbbox(self, _position, _text, font=None):
        assert font is not None
        return (0, 0, 12, 8)

    def text(self, position, text, **kwargs):
        self.texts.append((position, text, kwargs))


def test_image_descriptions_requires_api_key_for_openai_model():
    with pytest.raises(ValueError, match="API key required for OpenAI models"):
        smer.image_descriptions("gpt-4o-mini", "images")


def test_image_descriptions_openai_success_and_error(monkeypatch):
    files = [("one.png", "cat"), ("two.png", "dog")]
    monkeypatch.setattr(smer, "_get_image_files_with_class", lambda _path: iter(files))
    monkeypatch.setattr(smer, "_encode_image", lambda path: f"encoded:{path}")

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["messages"][0]["content"][1]["image_url"]["url"].endswith("two.png"):
                raise RuntimeError("boom")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="openai caption"))]
            )

    client = FakeClient()
    monkeypatch.setattr(smer, "OpenAI", lambda api_key: client)

    result = smer.image_descriptions("gpt-4o-mini", "images", api_key="secret")

    assert result["one.png"] == {
        "label": "cat",
        "description": "openai caption",
        "error": None,
    }
    assert result["two.png"]["label"] == "dog"
    assert result["two.png"]["description"] is None
    assert result["two.png"]["error"] == "boom"
    assert client.calls[0]["max_tokens"] == 20


def test_image_descriptions_local_model_init_error(monkeypatch):
    monkeypatch.setattr(
        smer.AutoProcessor,
        "from_pretrained",
        classmethod(lambda cls, _model: (_ for _ in ()).throw(RuntimeError("init failed"))),
    )

    with pytest.raises(ValueError, match="Error initializing local model: init failed"):
        smer.image_descriptions("local-model", "images")


def test_image_descriptions_local_model_success_and_per_file_error(monkeypatch):
    files = [("good.png", "cat"), ("bad.png", "dog")]
    opened = []
    cache_cleared = []

    monkeypatch.setattr(smer, "_get_image_files_with_class", lambda _path: iter(files))
    monkeypatch.setattr(smer.AutoProcessor, "from_pretrained", classmethod(lambda cls, _model: FakeProcessor()))
    monkeypatch.setattr(smer.AutoTokenizer, "from_pretrained", classmethod(lambda cls, _model: FakeTokenizer()))
    monkeypatch.setattr(
        smer.MllamaForConditionalGeneration,
        "from_pretrained",
        classmethod(lambda cls, *_args, **_kwargs: FakeLocalCaptionModel()),
    )
    monkeypatch.setattr(smer.torch.cuda, "empty_cache", lambda: cache_cleared.append(True))

    def fake_open(path):
        opened.append(path)
        if path == "bad.png":
            raise RuntimeError("cannot open")
        return FakeImage()

    monkeypatch.setattr(smer.Image, "open", fake_open)

    result = smer.image_descriptions("local-model", "images")

    assert result["good.png"]["description"] == "local caption"
    assert result["good.png"]["error"] is None
    assert result["bad.png"]["description"] is None
    assert result["bad.png"]["error"] == "cannot open"
    assert opened == ["good.png", "bad.png"]
    assert cache_cleared == [True]


def test_embed_descriptions_requires_api_key_for_openai_model():
    with pytest.raises(ValueError, match="API key required for OpenAI embeddings"):
        smer.embed_descriptions({}, "text-embedding-3-small")


def test_embed_descriptions_openai_success_error_and_missing_description(monkeypatch):
    descriptions = {
        "one.png": {"description": "red fox", "label": "cat"},
        "two.png": {"description": "bad", "label": "dog"},
        "three.png": {"description": None, "label": "bird"},
    }

    class FakeClient:
        def __init__(self):
            self.embeddings = self

        def create(self, input, model):
            assert model == "text-embedding-3-small"
            if input == "bad":
                raise RuntimeError("embed failed")
            return SimpleNamespace(data=[SimpleNamespace(embedding=[len(input)])])

    monkeypatch.setattr(smer, "OpenAI", lambda api_key: FakeClient())

    df = smer.embed_descriptions(descriptions, "text-embedding-3-small", api_key="secret")

    assert list(df.columns) == ["image", "description", "embedding", "label"]
    assert df.loc[df["image"] == "one.png", "embedding"].item() == [[3], [3]]
    assert df.loc[df["image"] == "two.png", "embedding"].item() is None
    assert df.loc[df["image"] == "three.png", "embedding"].item() is None


def test_embed_descriptions_local_model_init_error(monkeypatch):
    monkeypatch.setattr(
        smer.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, _model: (_ for _ in ()).throw(RuntimeError("no tokenizer"))),
    )

    with pytest.raises(ValueError, match="Error initializing local model: no tokenizer"):
        smer.embed_descriptions({}, "local-model")


def test_embed_descriptions_local_model_success_and_error(monkeypatch):
    descriptions = {
        "one.png": {"description": "hello world", "label": "cat"},
        "two.png": {"description": "bad token", "label": "dog"},
        "three.png": {"description": None, "label": "bird"},
    }

    monkeypatch.setattr(smer.AutoTokenizer, "from_pretrained", classmethod(lambda cls, _model: FakeTokenizer()))
    monkeypatch.setattr(smer.AutoModel, "from_pretrained", classmethod(lambda cls, _model: FakeLocalEmbeddingModel()))

    df = smer.embed_descriptions(descriptions, "local-model")

    first_embedding = df.loc[df["image"] == "one.png", "embedding"].item()
    assert len(first_embedding) == 2
    assert np.allclose(first_embedding[0], np.array([1.0, 2.0, 3.0]))
    assert df.loc[df["image"] == "two.png", "embedding"].item() is None
    assert df.loc[df["image"] == "three.png", "embedding"].item() is None


def test_aggregate_embeddings_returns_mean_vector():
    result = smer.aggregate_embeddings([[1, 3], [3, 5]])

    assert np.allclose(result, np.array([2.0, 4.0]))


def test_predict_proba_for_text_multiclass_uses_softmax():
    row = pd.Series(
        {
            "description": "red blue",
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        }
    )

    probs = smer._predict_proba_for_text("red", row, FakeMulticlassModel())

    expected = np.array([np.exp(1.0) / (np.exp(1.0) + 1.0), 1.0 / (np.exp(1.0) + 1.0)])
    assert np.allclose(probs, expected)


def test_predict_proba_for_text_binary_uses_zero_vector_when_no_words_match():
    row = pd.Series(
        {
            "description": "red blue",
            "embedding": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        }
    )

    probs = smer._predict_proba_for_text("missing", row, FakeBinaryModel())

    assert np.allclose(probs, np.array([0.5, 0.5]))


def test_classify_lr_builds_importance_columns_and_aopc_slice(monkeypatch):
    dataset = pd.DataFrame(
        {
            "description": ["red blue", "", "solo"],
            "embedding": [
                [np.array([3.0, 0.0]), np.array([1.0, 0.0])],
                [np.array([0.0, 0.0])],
                [np.array([2.0, 0.0])],
            ],
            "label": ["cat", "dog", "cat"],
        }
    )
    monkeypatch.setattr(smer, "_preprocess_text", lambda text: f"processed:{text}")

    df_aopc, updated = smer.classify_lr(dataset.copy(), np.zeros((1, 2)), FakePredictProbaModel())

    assert updated.loc[0, "feature_importance"].startswith("red:")
    assert updated.loc[0, "sorted_words_by_importance"] == "red,blue"
    assert updated.loc[1, "feature_importance"] == ""
    assert updated.loc[1, "sorted_words_by_importance_processed"] == "processed:"
    assert updated.loc[2, "sorted_words_by_importance"] == "solo"
    assert list(df_aopc["description"]) == ["solo"]


def test_compute_aopc_returns_average_probability_drops():
    df = pd.DataFrame(
        {
            "description": ["red blue"],
            "embedding": [[np.array([2.0, 0.0]), np.array([1.0, 0.0])]],
        }
    )

    scores = smer.compute_aopc(df, ["missing", "red"], 1, FakeBinaryModel())

    assert scores[0] == 0.0
    assert scores[1] > 0.0


def test_build_custom_predict_handles_present_and_missing_words():
    row = pd.Series(
        {
            "description": "red blue",
            "embedding": [np.array([2.0, 0.0]), np.array([0.0, 2.0])],
        }
    )
    model = FakePredictProbaModel()

    predict = smer.build_custom_predict(row, model)
    result = predict(["red", "missing"])

    assert result.shape == (2, 2)
    assert np.allclose(result[0], np.array([0.2, 0.8]))
    assert np.allclose(result[1], np.array([0.0, 1.0]))


def test_plot_aopc_uses_label_column_and_plots(monkeypatch):
    df = pd.DataFrame(
        {
            "description": ["solo", "red blue"],
            "embedding": [[np.array([1.0])], [np.array([1.0]), np.array([2.0])]],
            "label": ["cat", "dog"],
        }
    )
    compute_calls = []
    plot_calls = []

    monkeypatch.setattr(smer, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(
        smer,
        "_predict_proba_for_text",
        lambda text, _row, _model: np.array([0.9, 0.1]) if len(text.split()) >= 2 else np.array([0.6, 0.4]),
    )
    monkeypatch.setattr(smer, "build_custom_predict", lambda row, _model: f"predict:{row['description']}")
    monkeypatch.setattr(
        smer,
        "compute_aopc",
        lambda _df, top_words, max_k, _model: compute_calls.append((top_words, max_k)) or [0.0, 0.1],
    )

    class FakeExplainer:
        def __init__(self, class_names, random_state):
            assert class_names == ["cat", "dog"]
            assert random_state == 42

        def explain_instance(self, text_instance, classifier_fn, num_features):
            assert classifier_fn == f"predict:{text_instance}"
            assert num_features == len(text_instance.split())
            return SimpleNamespace(as_list=lambda: [("blue", -0.3), ("solo", 0.1)])

    monkeypatch.setattr(smer, "LimeTextExplainer", FakeExplainer)
    monkeypatch.setattr(smer.plt, "figure", lambda **kwargs: plot_calls.append(("figure", kwargs)))
    monkeypatch.setattr(smer.plt, "plot", lambda *args, **kwargs: plot_calls.append(("plot", args, kwargs)))
    monkeypatch.setattr(smer.plt, "xlabel", lambda label: plot_calls.append(("xlabel", label)))
    monkeypatch.setattr(smer.plt, "ylabel", lambda label: plot_calls.append(("ylabel", label)))
    monkeypatch.setattr(smer.plt, "title", lambda label: plot_calls.append(("title", label)))
    monkeypatch.setattr(smer.plt, "grid", lambda enabled: plot_calls.append(("grid", enabled)))
    monkeypatch.setattr(smer.plt, "legend", lambda: plot_calls.append(("legend", None)))
    monkeypatch.setattr(smer.plt, "show", lambda: plot_calls.append(("show", None)))

    smer.plot_aopc(df, FakeBinaryModel(), max_k=1)

    assert compute_calls == [(["blue", "red", "solo"], 1), (["blue", "solo"], 1)]
    assert ("show", None) in plot_calls


def test_plot_aopc_without_label_column_uses_empty_class_names(monkeypatch):
    df = pd.DataFrame(
        {
            "description": ["solo"],
            "embedding": [[np.array([1.0])]],
        }
    )

    monkeypatch.setattr(smer, "tqdm", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(smer, "_predict_proba_for_text", lambda *_args, **_kwargs: np.array([0.6, 0.4]))
    monkeypatch.setattr(smer, "build_custom_predict", lambda *_args, **_kwargs: "predict")
    monkeypatch.setattr(smer, "compute_aopc", lambda *_args, **_kwargs: [0.0])
    monkeypatch.setattr(smer.plt, "figure", lambda **_kwargs: None)
    monkeypatch.setattr(smer.plt, "plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(smer.plt, "xlabel", lambda _label: None)
    monkeypatch.setattr(smer.plt, "ylabel", lambda _label: None)
    monkeypatch.setattr(smer.plt, "title", lambda _label: None)
    monkeypatch.setattr(smer.plt, "grid", lambda _enabled: None)
    monkeypatch.setattr(smer.plt, "legend", lambda: None)
    monkeypatch.setattr(smer.plt, "show", lambda: None)

    class FakeExplainer:
        def __init__(self, class_names, random_state):
            assert class_names == []
            assert random_state == 42

        def explain_instance(self, **_kwargs):
            return SimpleNamespace(as_list=lambda: [("solo", 0.1)])

    monkeypatch.setattr(smer, "LimeTextExplainer", FakeExplainer)

    smer.plot_aopc(df, FakeBinaryModel(), max_k=0)


def test_plot_important_words_counts_and_returns_top_words(monkeypatch):
    dataset = pd.DataFrame(
        {
            "sorted_words_by_importance_processed": ["Alpha,beta", "", "gamma,delta", "alpha"],
        }
    )
    calls = []

    monkeypatch.setattr(smer.plt, "figure", lambda **kwargs: calls.append(("figure", kwargs)))
    monkeypatch.setattr(smer.sns, "barplot", lambda **kwargs: calls.append(("barplot", kwargs)))
    monkeypatch.setattr(smer.plt, "title", lambda value: calls.append(("title", value)))
    monkeypatch.setattr(smer.plt, "xlabel", lambda value: calls.append(("xlabel", value)))
    monkeypatch.setattr(smer.plt, "ylabel", lambda value: calls.append(("ylabel", value)))
    monkeypatch.setattr(smer.plt, "xticks", lambda **kwargs: calls.append(("xticks", kwargs)))
    monkeypatch.setattr(smer.plt, "tight_layout", lambda: calls.append(("tight_layout", None)))
    monkeypatch.setattr(smer.plt, "show", lambda: calls.append(("show", None)))

    top_words = smer.plot_important_words(dataset)

    assert list(top_words["word"]) == ["alpha", "gamma"]
    assert list(top_words["count"]) == [2, 1]
    assert any(name == "barplot" for name, _payload in calls)


def test_bounding_boxes_draws_labels_with_custom_font(monkeypatch):
    fake_image = FakeImage()
    fake_draw = FakeDraw()

    monkeypatch.setattr(smer.AutoProcessor, "from_pretrained", classmethod(lambda cls, _model: FakeProcessor()))
    monkeypatch.setattr(
        smer.AutoModelForZeroShotObjectDetection,
        "from_pretrained",
        classmethod(lambda cls, _model: SimpleNamespace(to=lambda _device: SimpleNamespace(__call__=lambda **_inputs: object()))),
    )
    monkeypatch.setattr(smer.Image, "open", lambda _path: fake_image)
    monkeypatch.setattr(smer.ImageDraw, "Draw", lambda _image: fake_draw)
    monkeypatch.setattr(smer.ImageFont, "truetype", lambda _font, _size: "font")

    class FakeModel:
        def to(self, _device):
            return self

        def __call__(self, **_inputs):
            return object()

    monkeypatch.setattr(
        smer.AutoModelForZeroShotObjectDetection,
        "from_pretrained",
        classmethod(lambda cls, _model: FakeModel()),
    )

    result = smer._bounding_boxes("image.png", pd.DataFrame({"word": ["cat"]}), "model")

    assert result is fake_image
    assert len(fake_draw.rectangles) == 2
    assert fake_draw.texts[0][1] == "cat: 0.95"


def test_bounding_boxes_falls_back_to_default_font(monkeypatch):
    fake_image = FakeImage()
    fake_draw = FakeDraw()
    load_default_calls = []

    monkeypatch.setattr(smer.AutoProcessor, "from_pretrained", classmethod(lambda cls, _model: FakeProcessor()))

    class FakeModel:
        def to(self, _device):
            return self

        def __call__(self, **_inputs):
            return object()

    monkeypatch.setattr(
        smer.AutoModelForZeroShotObjectDetection,
        "from_pretrained",
        classmethod(lambda cls, _model: FakeModel()),
    )
    monkeypatch.setattr(smer.Image, "open", lambda _path: fake_image)
    monkeypatch.setattr(smer.ImageDraw, "Draw", lambda _image: fake_draw)
    monkeypatch.setattr(
        smer.ImageFont,
        "truetype",
        lambda _font, _size: (_ for _ in ()).throw(IOError("missing font")),
    )
    monkeypatch.setattr(smer.ImageFont, "load_default", lambda: load_default_calls.append(True) or "default-font")

    smer._bounding_boxes("image.png", pd.DataFrame({"word": ["cat"]}), "model")

    assert load_default_calls == [True]


def test_save_bounding_box_images_processes_directory_and_returns_results(tmp_path, monkeypatch, capsys):
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "one.png").write_bytes(b"1")
    (input_dir / "two.jpg").write_bytes(b"2")
    (input_dir / "skip.txt").write_text("skip")

    images = {}

    def fake_bounding_boxes(image_path, **_kwargs):
        image = FakeImage()
        images[image_path] = image
        return image

    monkeypatch.setattr(smer, "_bounding_boxes", fake_bounding_boxes)

    result = smer.save_bounding_box_images(input_dir, output_dir, pd.DataFrame({"word": ["cat"]}))
    output = capsys.readouterr().out

    assert set(result.keys()) == {str(input_dir / "one.png"), str(input_dir / "two.jpg")}
    assert result[str(input_dir / "one.png")].endswith("one_annotated.png")
    assert images[str(input_dir / "one.png")].saved_paths == [str(output_dir / "one_annotated.png")]
    assert "Completed! Processed 2 images." in output


def test_save_bounding_box_images_handles_single_file_errors(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "image.png"
    output_dir = tmp_path / "out"
    image_path.write_bytes(b"img")

    monkeypatch.setattr(
        smer,
        "_bounding_boxes",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("detector failed")),
    )

    result = smer.save_bounding_box_images(image_path, output_dir, pd.DataFrame({"word": ["cat"]}))
    output = capsys.readouterr().out

    assert result == {str(image_path): None}
    assert "Error processing" in output
    assert "Completed! Processed 1 images." in output
