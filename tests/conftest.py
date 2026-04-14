import contextlib
import os
import sys
import types


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _install_dependency_stubs():
    spacy = types.ModuleType("spacy")
    spacy.load = lambda _name: (lambda _text: [])
    spacy.cli = types.SimpleNamespace(download=lambda _name: None)
    sys.modules["spacy"] = spacy

    torch = types.ModuleType("torch")

    @contextlib.contextmanager
    def no_grad():
        yield

    torch.no_grad = no_grad
    torch.bfloat16 = "bfloat16"
    torch.device = lambda name: name
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    sys.modules["torch"] = torch

    transformers = types.ModuleType("transformers")

    class _AutoTokenizerPlaceholder:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise NotImplementedError("Tests should monkeypatch this dependency.")

    class _AutoModelPlaceholder:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise NotImplementedError("Tests should monkeypatch this dependency.")

    class _MllamaPlaceholder:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise NotImplementedError("Tests should monkeypatch this dependency.")

    class _AutoProcessorPlaceholder:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise NotImplementedError("Tests should monkeypatch this dependency.")

    class _AutoObjectDetectionPlaceholder:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise NotImplementedError("Tests should monkeypatch this dependency.")

    transformers.AutoTokenizer = _AutoTokenizerPlaceholder
    transformers.AutoModel = _AutoModelPlaceholder
    transformers.MllamaForConditionalGeneration = _MllamaPlaceholder
    transformers.AutoProcessor = _AutoProcessorPlaceholder
    transformers.AutoModelForZeroShotObjectDetection = _AutoObjectDetectionPlaceholder
    sys.modules["transformers"] = transformers

    lime = types.ModuleType("lime")
    lime_text = types.ModuleType("lime.lime_text")

    class _PlaceholderLimeTextExplainer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def explain_instance(self, *args, **kwargs):
            raise NotImplementedError("Tests should monkeypatch this dependency.")

    lime_text.LimeTextExplainer = _PlaceholderLimeTextExplainer
    sys.modules["lime"] = lime
    sys.modules["lime.lime_text"] = lime_text


_install_dependency_stubs()
