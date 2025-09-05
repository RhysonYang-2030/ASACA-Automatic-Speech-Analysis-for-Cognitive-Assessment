"""
Classifier + SHAP explainability for ASACA cognition add-on.
"""

from pathlib import Path
import joblib, numpy as np, shap, matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from asaca_cognition.feature_extractor import (
    extract_features,
    CognitionDictionaryBank,
    SCALAR_KEYS,
    DICT_FEATS,
    build_feature_vector,
)

FEATURE_NAMES = SCALAR_KEYS + DICT_FEATS          # 17-element list

class CognitionClassifier:
    """
    • predict_label(wav)     → 'HC' / 'AD' / 'MCI'
    • explain(wav, class)    → [(feat, shap_val), …]  and optional PNG
    """
    def __init__(
        self,
        model_pkl: Path,
        dict_dir: Path,
        processor_path: str,
        asr_model_path: str,
        device: str = "cpu",
        feature_xlsx: Path | None = None,   # background data for SHAP
        bg_size: int = 120                  # how many samples for explainer
    ):
        # --- load model & resources -----------------------------------
        self.pipe = joblib.load(model_pkl)          # Pipeline (scaler + LR)
        self.bank = CognitionDictionaryBank(dict_dir)
        self.proc = Wav2Vec2Processor.from_pretrained(processor_path)
        self.asr  = Wav2Vec2ForCTC.from_pretrained(asr_model_path).to(device).eval()
        self.device = device

        # --- build SHAP explainer (optional) --------------------------
        self.explainer = None
        if feature_xlsx and Path(feature_xlsx).exists():
            import pandas as pd, json
            df = pd.read_excel(feature_xlsx)
            # sample N lines to keep explainer light
            X_bg = np.vstack(
                df.sample(n=min(bg_size, len(df)), random_state=1)
                  ["vector"].apply(lambda s: json.loads(s)).values
            )
            self.explainer = shap.LinearExplainer(
                self.pipe.named_steps["clf"],   # final LR estimator
                X_bg,
                feature_perturbation="independent"
            )

    # ------------------------------------------------------------------
    def predict_label(
        self,
        wav: Path | None = None,
        *,
        transcription: str | None = None,
        global_feats: dict | None = None,
    ) -> str:
        """Predict cognitive label from a WAV or pre-computed features."""
        if transcription is not None and global_feats is not None:
            vec, _ = build_feature_vector(transcription, global_feats, self.bank)
        else:
            if wav is None:
                raise ValueError("wav or transcription+global_feats required")
            vec, _ = extract_features(wav, self.asr, self.proc, self.bank, self.device)

        pred_int = int(self.pipe.predict(vec.reshape(1, -1))[0])
        return {0: "HC", 1: "AD", 2: "MCI"}[pred_int]

    # ------------------------------------------------------------------
    def explain(
        self,
        wav: Path | None = None,
        class_name: str = "HC",
        save_png: Path | None = None,
        *,
        transcription: str | None = None,
        global_feats: dict | None = None,
    ):
        """
        Returns list[(feature, shap_value)] for the selected class.
        If save_png is provided, also writes a horizontal bar plot.
        """
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not initialised "
                               "(check feature_xlsx path).")

        class_idx = {"HC": 0, "AD": 1, "MCI": 2}[class_name]
        if transcription is not None and global_feats is not None:
            vec, _ = build_feature_vector(transcription, global_feats, self.bank)
        else:
            if wav is None:
                raise ValueError("wav or transcription+global_feats required")
            vec, _ = extract_features(wav, self.asr, self.proc, self.bank, self.device)
        vals = self.explainer.shap_values(vec)[class_idx]   # ndarray (17,)

        if save_png:
            plt.figure(figsize=(6, 4))
            shap.plots.bar(
                shap.Explanation(
                    values=vals,
                    base_values=None,
                    data=vec,
                    feature_names=FEATURE_NAMES
                ),
                max_display=17,
                show=False
            )
            plt.title(f"SHAP – contribution toward “{class_name}”")
            plt.tight_layout()
            plt.savefig(save_png, dpi=300)
            plt.close()

        return list(zip(FEATURE_NAMES, vals))

