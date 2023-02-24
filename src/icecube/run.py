import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from icecube.pipeline import train


@hydra.main(version_base=None, config_path="../../conf/", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    if cfg.train:
        train(cfg)


if __name__ == "__main__":
    main()
