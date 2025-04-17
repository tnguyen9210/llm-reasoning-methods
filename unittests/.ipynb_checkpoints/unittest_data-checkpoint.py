
from datasets import Dataset, load_dataset

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score


def main():
    parser = H4ArgumentParser(Config)
    print(parser)
    config = parser.parse()
    print(config)

    # base_dir
    base_dir = '/groups/kjun/tnn/datasets/'
    
    # dataset path
    data_dir = base_dir + "/math500"
    
    dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        cache_dir=data_dir
    )

    print(dataset)
    
        
    

if __name__ == "__main__":
    main()
