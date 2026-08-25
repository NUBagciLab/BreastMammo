# [MICCAI2026 Workshop Deep-Brea3th] BreastMammo and DenseMammo: Benchmarks for Mammography Domain Generalization 
<p align="center">
  <img src="figures/samples.png" alt="" width="100%" />
</p>

<p align="center">
  <img src="figures/pipeline.png" alt="" width="100%" />
</p>

# Paper link:

arXiv: https://arxiv.org/abs/2608.10271

# Dataset:

BreastMammo: https://osf.io/n4yr2/

DenseMammo: https://osf.io/4azcr/

# Pre-trained Weights:
https://huggingface.co/phy710/BreastMammo

# Generative Synthetic Images
You may run histogram.py or dft.py for histogram-based and DFT-based domain generation. You may revise the following code at lines 55--61 in histogram.py and lines 99--105 in dft.py to choose the source and reference dataset.

To generate synthetic BreastMammo images in the DenseMammo domain:

    source_root = BreastMammo_root
    source = BreastMammo_density(root = BreastMammo_root)
    ref = DenseMammo_density(root= DenseMammo_root)


To generate synthetic DenseMammo images in the BreastMammo domain:

    source_root = DenseMammo_root
    ref = BreastMammo_density(root = BreastMammo_root)
    source = DenseMammo_density(root= DenseMammo_root)

# Training and Testing
In each task, go to the corresponding folder then run

    ./main.sh [-model model_name] [-input_size size] [-data_path data_path]

for CNNs. Here, [-input_size] can be 224 or 512, [-model] can be efficientnet_b0, densenet121, resnet50.

    ./main_swin.sh [-input_size size] [-data_path data_path]

for swin-T. Other models may be supported but not tested yet.
For example:

    ./main.sh -model efficientnet_b0 -input_size 224 -data_path /dataset/LUMINA_PNG

You can get the test results by running the command like the following:

    python fold_test.py --model --data-path /dataset/LUMINA_PNG --model efficientnet_b0 --input-size 224
Here, [--input-size] can be 224 or 512, [--model] can be efficientnet_b0, densenet121, resnet50, or swin_t.

The pretrained weights are available at https://huggingface.co/phy710/BreastMammo

# Citation
If you use this dataset in your research, please cite our MICCAI paper:

    Hongyi Pan, Gorkem Durak, Halil Ertugrul Aktas, Andrea Mia Bejar, Mustafa Ege Seker, Nebile Alibeyoglu, Rumeysa Guclu, Rana Gunoz Comert Bozkurt, Sibel Ozkan Gurdal, Neslihan Cabioglu, Beyza Ozcinar, Ravza Yilmaz, Vahit Ozmen, Erkin Aribal, Sukru Mehmet Erturk, Yalda Zafari, Mohamed Mabrok, Kayhan Batmanghelich, Mohammad Yaqub, Ziyue Xu, Ulas Bagci. “BreastMammo and DenseMammo: Benchmarks for Mammography Domain Generalization.” MICCAI 2026.
