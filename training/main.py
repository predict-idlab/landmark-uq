import sys

sys.path.append("../src")

import landmarker  # type: ignore
import lightning as L
import monai
import numpy
from datasets.bcg import BCGLightningDataModule
from datasets.isbi2015 import ISBI2015LightningDataModule
from datasets.mml import MLLLightningDataModule
from datasets.oai_pelvis import OAIPelvisLightningDataModule
from lightning.pytorch.cli import LightningCLI
from models import *
from torch.utils.data import DataLoader

cli = LightningCLI(seed_everything_default=1817)  # pass all args to LightningCLI
