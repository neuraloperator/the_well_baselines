from pathlib import Path
import argparse

try:
    from the_well.utils.download import well_download
except:
    print("You are trying to use WellDataset without optional dependency `the_well`. ",
          "Install via `pip install the_well` and retry.")
    raise ImportError


class TheWellDataset:
    """Download one dataset from The Well [1]_.

        Files land in root_dir/datasets/<well_dataset_name>/: stats.yaml and
        data/{train,valid,test}/*.hdf5.
        Parameters
        ----------
        root_dir : Path
            directory under which the datasets/ folder is created
        well_dataset_name : str
            name of the dataset to download
        download : bool, optional
            whether to download missing splits, by default True
        first_only : bool, optional
            fetch only the first HDF5 file of each split, by default True

        References
        ----------
        .. [1] : Ohana, R., McCabe, M., Meyer, L., Morel, R., Agocs, F., Beneitez, M., Berger, M.,
            Burkhart, B., Dalziel, S., Fielding, D., Fortunato, D., Goldberg, J., Hirashima, K., Jiang, Y.,
            Kerswell, R., Maddu, S., Miller, J., Mukhopadhyay, P., Nixon, S., Shen, J., Watteaux, R.,
            Regaldo-Saint Blancard, B., Rozet, F., Parker, L., Cranmer, M., and Ho, S. (2024).
            The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning.
            NeurIPS 2024, https://openreview.net/forum?id=00Sx577BT3.
        """
    def __init__(self,
                 root_dir: Path, 
                 well_dataset_name : str,
                 download: bool=True,
                 first_only: bool=True,
                 ):
        
        base_path = root_dir / f"datasets/{well_dataset_name}/data"

        if download:
            for split in ['train', 'test', 'valid']:
                data_path = base_path / split
                if not data_path.exists():
                    well_download(root_dir,
                                dataset=well_dataset_name,
                                split=split,
                                first_only=first_only,
                                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--well_dataset_name", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Downloading {args.well_dataset_name} Dataset...")
    
    dataset = TheWellDataset(Path(args.root_dir), args.well_dataset_name, 
                             download=True,
                             first_only=False)
    
    print(f"{args.well_dataset_name} Dataset downloaded")
