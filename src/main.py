import logging
import hydra
import ssl
from omegaconf import DictConfig, OmegaConf

# macOS SSL fix for torchvision downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from src.utils.seed import seed_everything
from src.utils.hardware import get_device
from src.utils.logger import MetricsLoggerCallback

from src.data.cifar_datamodule import get_dataloaders, get_numpy_data
from src.models.classical import build_svm_pipeline
from src.models.deep_learning import build_dl_model
from src.engine.trainer_ml import train_and_evaluate_ml
from src.engine.trainer_dl import train_and_evaluate_dl
from src.ui.dashboard import RichDashboardCallback

log = logging.getLogger(__name__)

def count_parameters(model):
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0 # SVM has no dynamic parameter counting here

@hydra.main(version_base="1.3", config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 1. Setup
    seed_everything(cfg.seed)
    device = get_device()
    log.info(f"Hardware device selected: {device}")
    
    model_name = cfg.model.name
    
    # 2. Branch: ML vs DL
    if model_name == "svm":
        log.info("Preparing data for Classical ML...")
        X_train, y_train, X_test, y_test = get_numpy_data(cfg.data)
        
        if cfg.debug_mode:
            log.info("DEBUG MODE: Using subset of data")
            X_train, y_train = X_train[:500], y_train[:500]
            X_test, y_test = X_test[:100], y_test[:100]
            
        pipeline = build_svm_pipeline(cfg.model)
        
        logger = MetricsLoggerCallback(model_name=model_name, param_count=0)
        train_and_evaluate_ml(pipeline, X_train, y_train, X_test, y_test, callbacks=[logger])
        
    elif model_name in ["resnet18", "vit_b16"]:
        log.info("Preparing data for Deep Learning...")
        trainloader, testloader = get_dataloaders(cfg.data, model_name=model_name)
        
        model = build_dl_model(cfg.model)
        param_count = count_parameters(model)
        
        # Override epochs if debug mode
        if cfg.debug_mode:
            log.info("DEBUG MODE: Limiting to 1 epoch")
            cfg.model.epochs = 1
            
        # Instantiate the Callbacks
        dashboard = RichDashboardCallback(model_name=model_name)
        logger = MetricsLoggerCallback(model_name=model_name, param_count=param_count)
            
        try:
            train_and_evaluate_dl(model, trainloader, testloader, device, cfg.model, callbacks=[dashboard, logger])
        except KeyboardInterrupt:
            dashboard.live.stop()
            log.warning("🛑 Training interrupted by user.")
        except Exception as e:
            dashboard.live.stop()
            log.error(f"Training failed: {e}")
            raise e
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")

if __name__ == "__main__":
    main()