import hydra
from omegaconf import DictConfig

from icecube.pipeline import fit_one_cycle


@hydra.main(version_base=None, config_path="../../conf/", config_name="train")
def main(cfg: DictConfig):
    pipeline = {
        "train": fit_one_cycle,
    }
    
    pipeline[cfg.pipeline](cfg)

if __name__ == "__main__":
    main()