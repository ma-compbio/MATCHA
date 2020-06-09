# MATCHA: Probing multi-way chromatin interaction with hypergraph representation learning

This is the implementation of the algorithm MATCHA for analyzing multi-way chromatin interaction data via hypergraph representation learning.

## Requirements
The main part of the alogrithm (`process.py, generate_kmers.py, main.py`) requires


- h5py
- numpy
- pytorch
- pybloom_live (https://github.com/joseph-fox/python-bloomfilter)
- scikit-learn
- tqdm


The visualization part of the algorihtm (`denoise_contact.py`) requires

- seaborn
- matplotlib

## Configure the parameters

All the input parameters are stored in the config.JSON file. 
Please fill in this file before running the program.
Note that, some scripts only use part of these parameters, so these parameters can be filled in before running those specific script.

| params       | description                  | example                   | used in    |
|--------------|------------------------------|---------------------------|------------|
| cluster_path | the path of the cluster file | "./4DNFIBEVVTN5.clusters" | process.py |
|mcool_path | the path of the mcool file | "./4DNFIUOOYQC3.mcool" | process.py|
|resolution | the resolution to consider (bin size) | 1000000 | process.py|
|chrom_list | list of the chromosomes to consider | ["chr1", "chr2"] | process.py, main.py|
|chrom_size | the path of the chromatin size file | "./hg38.chrom.sizes.txt" | process.py|
|temp_dir | the directory of the temp files to store | "../Temp" | all|
|max_cluster_size| the maximum cluster size to consider | 25| process.py, generate_kmers.py |
|min_distance | minimum pairwise genomic distance constraint for multi-way interactions (in unit of the number of bins) |0| generate_kmers.py, main.py, denoise_contact.py|
|k-mer_size| list of the size of the k-mers to considier | [2,3,4,5] | generate_kmers.py, main.py, 
|min_freq_cutoff | only consider k-mers with occurrence frequency >= | 2 | generate_kmers.py|
|quantile_cutoff_for_positive | the quantile cutoff of hyperedges to be considered as positive samples. For instance, 0.6 represents the hyperedges with occurrence frequency in the top 40% (>= 0.6) would be used as positive samples. The cut-off is applied to different sized hyperedges separately| 0.6 | main.py | 
|quantile_cutoff_for_unlabel | the quantile cutoff of hyperedges to be considered as non-negative samples (positive + samples that cannot be confidently classified as either positive or negative samples) | 0.4 | main.py | 
|embed_dim | embedding dimensions for the bins | 64| main.py|


## Usage
1. `cd Code`
2. Run `python process.py`, which will parse the input cluster file, mcool file and the chromosome size files. There will be 3 key output files:
   1. `bin2node.npy, node2bin.npy` within the `temp_dir` above. As the name indicates, it's a dictionary that maps the genomic bin to the node id and vice verse. The genomic bin has the format of `chr1:2000000`
   2. `node2chrom.npy`. It maps the node id to the chromosome.
   3. All these dictionaries can be loaded through `np.load(FILEPATH, allow_picke=True).item()`
3. Run `python generate_kmers.py`, which will further transfer the parsed cluster file into a list of k-mers (hyperedges) with the corresponding occurrence frequencies. The output files are
   1. `all_<k-mer size>_counter.npy`: the generated k-mers
   2. `all_<k-mer size>_freq_counter.npy`: the occurrence frequency corresponds to the generated k-mers
4. Run `python main.py`, which will train the model based on the generated dataset. The output includes:
   1. `model2load` within the `temp_dir` above. The model can be loaded by `model = torch.load(FILEPATH)`. The model can return predictions through `model(x)`. Note that the `x` should be a pytorch tensor of dtype `torch.long`
   2. `embeddings.npy` lies in the root dir. It's the embedding vectors for the genomic bins. The shape of the vectors are `(num of genomic bins, embed_dim chosen above)`. The mapping relationship between the genomic bin and its index in this vector can be retrived in the dictionary `node2bin.npy, bin2node.npy` mentioned above.
5. To generate the denoised contact matrix, run `python denoise_contact.py` There will be output figures named as `chr1_origin.png` and `chr1_denoise.png`, etc... produced in the root dir. There will also be an mcool file named as `denoised.mcool` in the root dir, which contains the denoised intra-chromosomal contact matrix at the given resolution.

6. To predict the probabilities of forming multi-way chromatin interactions for a custom list of genome coordinate, run `python predict_multiway.py -i INPUT_FILE -o OUTPUT_FILE`. The `INPUT_FILE` should be a text file where each line is a tab separated list of genome coordinates. For example:
```text
chr1:1000000<tab>chr2:20000<tab>chr3:40000
chr1:1000000<tab>chr2:20000<tab>chr3:40000<tab>chr1:12345
```
The output file will be a list of the probability scores stored in the `OUTPUT_FILE`

## Cite

If you want to cite our paper

```
@article{zhang2020matcha,
  title={MATCHA: Probing Multi-way Chromatin Interaction with Hypergraph Representation Learning},
  author={Zhang, Ruochi and Ma, Jian},
  journal={Cell Systems},
  volume={10},
  number={5},
  pages={397--407},
  year={2020},
  publisher={Elsevier}
}
```
