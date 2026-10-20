"""Fixed, text-only style features for the separately recorded v2 experiment.

Feature families follow the character/word/function-word measurements in
https://arxiv.org/html/2104.08530v2, extended with Unicode punctuation and
position-specific discourse features. This is not a reproduction of that paper.
"""

from __future__ import annotations

from collections import Counter
import re
import string
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC


# Fixed before fitting; no corpus vocabulary, labels, IDs, or prompt IDs.
FUNCTION_WORDS = tuple(dict.fromkeys("""
a an the and or but if because although though while whereas unless until when
whenever where wherever as than that whether since once before after during
despite through throughout within without with into onto upon from of for to
at by in on over under between among against about above below beyond across
around along toward towards per via not no nor never neither either both each
every any all some many much few little more most less least several enough
other another such same own only even also just still already yet again rather
quite very too so then thus hence therefore however instead otherwise indeed
perhaps maybe almost always often sometimes usually generally typically
especially particularly simply merely really actually essentially ultimately
is are was were be been being am do does did done doing have has had having
can could may might must shall should will would need ought
i me my mine myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs
themselves one ones someone anyone everyone nobody nothing something everything
this these those there here who whom whose which what why how
""".split()))

CONNECTIVES = (
    "for example", "for instance", "in other words", "in practice",
    "in fact", "as a result", "on the other hand", "at the same time",
    "rather than", "instead of", "not only", "but also", "even if",
    "even though", "as long as", "so that", "such as", "in contrast",
    "by contrast", "in short", "ultimately", "however", "therefore",
    "because", "although", "while", "unless", "whether", "if",
)
PUNCTUATION = tuple(dict.fromkeys(string.punctuation + "—–‘’“”…−•\n\t"))
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+(?:\.\d+)?", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+")


class StyleFeatures(TransformerMixin, BaseEstimator):
    """Deterministic measurements; fit deliberately learns nothing."""

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self._measure("").keys()), dtype=object)

    def transform(self, X):
        if isinstance(X, str):
            raise ValueError("Expected a sequence of passages, not one string.")
        rows = [list(self._measure(text).values()) for text in X]
        if not rows:
            return np.empty((0, len(self.get_feature_names_out())), dtype=np.float64)
        matrix = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite style feature.")
        return matrix

    @staticmethod
    def _measure(text):
        if not isinstance(text, str):
            raise TypeError("StyleFeatures accepts text strings only.")
        words = WORD_RE.findall(text.lower())
        counts = Counter(words)
        n = max(1, len(text))
        nw = max(1, len(words))
        lengths = np.asarray([len(w) for w in words] or [0], dtype=float)
        sentences = [s for s in SENTENCE_RE.split(text) if s.strip()]
        sentence_lengths = np.asarray(
            [len(WORD_RE.findall(s)) for s in sentences] or [0], dtype=float,
        )
        ns = max(1, len(sentences))
        f = {
            "log_chars": np.log1p(len(text)),
            "log_words": np.log1p(len(words)),
            "log_sentences": np.log1p(len(sentences)),
            "words_per_sentence": len(words) / ns,
            "chars_per_word": len(text) / nw,
            "type_token_ratio": len(counts) / nw,
            "hapax_ratio": sum(v == 1 for v in counts.values()) / nw,
            "repeated_token_ratio": sum(v for v in counts.values() if v > 1) / nw,
            "digits_rate": sum(c.isdigit() for c in text) / n,
            "uppercase_rate": sum(c.isupper() for c in text) / n,
            "letters_rate": sum(c.isalpha() for c in text) / n,
            "whitespace_rate": sum(c.isspace() for c in text) / n,
            "unicode_rate": sum(ord(c) > 127 for c in text) / n,
            "unicode_punctuation_rate": sum(
                ord(c) > 127 and unicodedata.category(c).startswith("P") for c in text
            ) / n,
            "contraction_rate": sum("'" in w or "’" in w for w in words) / nw,
            "short_word_rate": float(np.mean(lengths <= 3)),
            "long_word_rate": float(np.mean(lengths >= 8)),
            "question_sentence_rate": sum("?" in s for s in sentences) / ns,
            "exclamation_sentence_rate": sum("!" in s for s in sentences) / ns,
            "sentence_length_alternation": float(np.abs(np.diff(sentence_lengths)).mean())
            if len(sentence_lengths) > 1 else 0.,
        }
        for prefix, values in (("word_length", lengths), ("sentence_words", sentence_lengths)):
            for name, value in (
                ("mean", np.mean(values)), ("std", np.std(values)),
                ("min", np.min(values)), ("max", np.max(values)),
                ("q25", np.quantile(values, .25)), ("q50", np.quantile(values, .5)),
                ("q75", np.quantile(values, .75)),
            ):
                f[f"{prefix}_{name}"] = float(value)
        for length in range(1, 16):
            f[f"word_length_bin_{length}"] = float(np.mean(lengths == length))
        for char in PUNCTUATION:
            key = f"punct_u{ord(char):04x}"
            f[f"{key}_rate"] = text.count(char) / nw
            f[f"{key}_present"] = float(char in text)
        for char in string.ascii_lowercase:
            f[f"letter_{char}_rate"] = text.lower().count(char) / n
        for word in FUNCTION_WORDS:
            f[f"fw_{word}"] = counts[word] / nw
        lower = " " + " ".join(words) + " "
        first = " " + " ".join(words[:15]) + " "
        last = " " + " ".join(words[-15:]) + " "
        for connective in CONNECTIVES:
            needle = " " + connective + " "
            f[f"connective_{connective}"] = lower.count(needle) / nw
            f[f"opening_{connective}"] = float(needle in first)
            f[f"ending_{connective}"] = float(needle in last)
        for position, sentence in (
            ("opening", sentences[0] if sentences else ""),
            ("ending", sentences[-1] if sentences else ""),
        ):
            sw = WORD_RE.findall(sentence.lower())
            f[f"{position}_length_share"] = len(sw) / nw
            f[f"{position}_question"] = float("?" in sentence)
            f[f"{position}_colon"] = float(":" in sentence)
            f[f"{position}_dash"] = float("—" in sentence)
            f[f"{position}_first_person"] = sum(w in {"i", "we", "our", "my"} for w in sw) / max(1, len(sw))
            f[f"{position}_second_person"] = sum(w in {"you", "your"} for w in sw) / max(1, len(sw))
        return f


CANDIDATES = {
    "style_logistic": {
        "features": "fixed StyleFeatures -> StandardScaler",
        "classifier": "LogisticRegression", "C": .05,
        "solver": "lbfgs", "max_iter": 1500,
    },
    "style_hist": {
        "features": "fixed StyleFeatures", "classifier": "HistGradientBoostingClassifier",
        "max_iter": 160, "max_leaf_nodes": 7, "learning_rate": .07,
        "l2_regularization": 3., "min_samples_leaf": 25,
        "early_stopping": False, "random_state": 42,
    },
    "char_style_svm": {
        "features": "char TFIDF (3,5), min_df2, max_features250000, case-sensitive"
        " + .5 * L2-normalized standardized style; concatenate and L2-normalize",
        "classifier": "LinearSVC", "C": 1., "max_iter": 10000,
        "dual": True, "random_state": 42,
    },
}


def build_style_pipeline(name):
    style = StyleFeatures()
    if name == "style_logistic":
        return Pipeline([
            ("style", style), ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=.05, solver="lbfgs", max_iter=1500)),
        ])
    if name == "style_hist":
        return Pipeline([
            ("style", style),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=160, max_leaf_nodes=7, learning_rate=.07,
                l2_regularization=3., min_samples_leaf=25,
                early_stopping=False, random_state=42,
            )),
        ])
    if name == "char_style_svm":
        return Pipeline([
            ("features", FeatureUnion([
                ("char", TfidfVectorizer(
                    analyzer="char", ngram_range=(3, 5), min_df=2,
                    max_features=250_000, lowercase=False, strip_accents=None,
                    sublinear_tf=True, dtype=np.float32,
                )),
                ("style", Pipeline([
                    ("extract", style), ("scale", StandardScaler()),
                    ("normalize", Normalizer()),
                ])),
            ], transformer_weights={"char": 1., "style": .5})),
            ("normalize", Normalizer()),
            ("classifier", LinearSVC(
                C=1., max_iter=10000, dual=True, random_state=42, tol=1e-4,
            )),
        ])
    raise ValueError(name)

