from utils.edge_power import get_edge_power_json
from pathlib import Path

CAMO_PATH = Path('/content/drive/MyDrive/Camouflage/Images/camo/')
TRAIN_PATH = CAMO_PATH / 'Train'
TEST_PATH = CAMO_PATH / 'Test'

get_edge_power_json(TRAIN_PATH)
            