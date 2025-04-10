
from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    print(config)
    print(parser)


if __name__ == "__main__":
    main()
