import sys
import json
from pathlib import Path
import torch
from PIL import Image, ImageOps, ImageFile, UnidentifiedImageError
from transformers import ConvNextForImageClassification, ConvNextImageProcessor

from utils.logger import logging
from utils.exception import CustomException
from utils.read_yaml import read_yaml

from torchvision import transforms
from torchvision.transforms import InterpolationMode

CONFIG_PATH = Path("configuration-files/cv_inference_config.yaml")


class AgroLensPredictor:
    """Lightweight ConvNeXt inference wrapper that returns only the predicted label."""

    def __init__(self, model_path: Path, device: str = "cpu", class_names=None):
        try:
            requested = str(device).lower()
            if requested == "cuda" and not torch.cuda.is_available():
                logging.warning("CUDA requested but not available. Falling back to CPU.")
                requested = "cpu"
            if requested == "mps" and not getattr(torch.backends, "mps", None):
                requested = "cpu"
            elif requested == "mps" and not torch.backends.mps.is_available():
                logging.warning("MPS requested but not available. Falling back to CPU.")
                requested = "cpu"
            self.device = torch.device(requested)
            logging.info(f"Using device: {self.device}")
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            logging.info(f"Loading model and processor from: {model_path}")
            self.model = ConvNextForImageClassification.from_pretrained(str(model_path)).to(self.device)
            self.processor = ConvNextImageProcessor.from_pretrained(str(model_path))
            self.model.eval()
            logging.info("Model and processor loaded successfully.")
            self.eval_tf = self._build_eval_transform()

            if class_names is not None:
                self.id2label = {i: name for i, name in enumerate(class_names)}
                self.model.config.id2label = self.id2label
                logging.info("Custom class names loaded (from config file).")
            else:
                self.id2label = self.model.config.id2label
                logging.info("Using label mappings from model config.")
        except Exception as e:
            logging.error("Failed to initialize AgroLensPredictor.", exc_info=True)
            raise CustomException(e)

    def _build_eval_transform(self, size: int | None = None):
        """Build eval transform to match training eval (Resize->CenterCrop, Normalize)."""
        try:
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
        except Exception as e:
            logging.error("torchvision not available for eval transforms.", exc_info=True)
            raise CustomException(e)
        im_size = size or getattr(getattr(self.model, "config", object()), "image_size", 224)
        return transforms.Compose([
            transforms.Resize(int(im_size / 0.875), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std),
        ])

    def _open_image_safe(self, path: Path) -> Image.Image:
        """Open image safely and ensure RGB mode & orientation."""
        try:
            with Image.open(path) as im:
                try:
                    im = ImageOps.exif_transpose(im)
                except Exception:
                    pass
                if getattr(im, "is_animated", False):
                    im.seek(0)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                return im.copy()
        except (UnidentifiedImageError, OSError) as e:
            logging.error(f"Image open failed for: {path} ({e})", exc_info=True)
            raise CustomException(e)

    def __call__(self, image_path: Path) -> str:
        """Run inference and return the top predicted label."""
        try:
            image = self._open_image_safe(image_path)
            pixel = self.eval_tf(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(pixel_values=pixel).logits.squeeze(0)
                pred_idx = torch.argmax(logits).item()

            pred_label = self.id2label.get(pred_idx, f"LABEL_{pred_idx}")
            logging.info(f"Prediction successful. Predicted label: {pred_label}")
            return pred_label
        except Exception as e:
            logging.error("Prediction failed.", exc_info=True)
            raise CustomException(e)


def _load_class_names(path: Path | None):
    """Optionally load class names from JSON (list or {'class_names': [...]}) or TXT (one label per line)."""
    if path is None:
        return None
    try:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"class_names_path not found: {p}")
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "class_names" in data:
                return list(data["class_names"])
            if isinstance(data, list):
                return list(data)
            raise ValueError("JSON must be a list or contain key 'class_names'.")
        elif p.suffix.lower() in {".txt", ".labels"}:
            with open(p, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("Unsupported class_names file type. Use .json or .txt")
    except Exception as e:
        logging.error("Failed to load class names from file.", exc_info=True)
        raise CustomException(e)


def load_cfg():
    """Load YAML config (all values)."""
    cfg = read_yaml(CONFIG_PATH)
    logging.info(f"Loaded config from {CONFIG_PATH}")
    try:
        logging.setLevel(cfg.log_level)
    except Exception:
        pass
    return cfg


def build_predictor(cfg) -> AgroLensPredictor:
    """Build predictor using YAML config (model/device/class names)."""
    model_path = Path(cfg.model.model_path)
    device = cfg.model.device
    class_names_path = getattr(cfg.model, "class_names_path", None)

    if not model_path.exists():
        raise CustomException(FileNotFoundError(f"Model path does not exist: {model_path}"))

    class_names = _load_class_names(Path(class_names_path)) if class_names_path else None
    return AgroLensPredictor(model_path=model_path, device=device, class_names=class_names)


def run_inference(image_path: Path | None = None, cfg=None) -> str:
    """
    Run a single inference using YAML for everything except the image path.
    If image_path is None, fall back to cfg.image.input_path.
    Also handles optional saving of the predicted label JSON as per YAML.
    """
    if cfg is None:
        cfg = load_cfg()

    if image_path is None:
        if not hasattr(cfg, "image") or not getattr(cfg.image, "input_path", None):
            raise CustomException(ValueError("No image path provided and cfg.image.input_path missing."))
        img_path = Path(cfg.image.input_path)
    else:
        img_path = Path(image_path)

    try:
        supported_exts = {ext.lower() for ext in cfg.image.supported_formats}
    except Exception:
        supported_exts = {".jpg", ".jpeg", ".png", ".webp"}

    if not img_path.exists():
        raise CustomException(FileNotFoundError(f"Image path does not exist: {img_path}"))
    if img_path.suffix.lower() not in supported_exts:
        raise CustomException(
            ValueError(f"Unsupported image format '{img_path.suffix}'. Supported: {sorted(supported_exts)}"),
        )

    predictor = build_predictor(cfg)
    pred_label = predictor(img_path)

    save_enabled = bool(cfg.output.save_enabled)
    if save_enabled:
        save_dir = Path(cfg.output.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / cfg.output.file_name
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"predicted_label": pred_label}, f)
        logging.info(f"Prediction saved to {save_path}")

    logging.info(f"Predicted: {pred_label}")
    print(f"🧠 Predicted: {pred_label}")
    return pred_label


if __name__ == "__main__":
    try:
        IMG_PATH = Path(r"DATA/cv-inference-data/applescaborsomething.png")

        cfg = load_cfg()
        image_override = IMG_PATH if IMG_PATH is not None else None
        run_inference(image_override, cfg=cfg)

    except Exception as e:
        logging.error("AgroLensPredictor pipeline failed.", exc_info=True)
        raise CustomException(e)
