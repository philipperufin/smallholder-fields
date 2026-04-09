# smallholder-fields

This repository provides the code used to produce the MozFields 2023 dataset described in [Rufin et al. 2026](doi.org/10.1088/1748-9326/ae5cb4). 

The repository is currently in the final stages of perparation and will be released as soon as possible. The published code will include scripts for
- creating multi-task labels from imagery through [pseudo-labels](https://doi.org/10.1016/j.jag.2024.104149)
- FracTAL ResUNet model fine-tuning
- field delineation inference
- post-processing incl. polygonization, merging, ML-based error cleaning 

The following resources are already available: 

- [GitHub repository containing code for producing pseudo-labels](https://github.com/philipperufin/pseudo-fields)
- [FracTAL ResUNet Model checkpoint](https://doi.org/10.5281/zenodo.17531365)
- [National scale cropland fraction and field size metrics at 0.05° resolution](https://doi.org/10.5281/zenodo.18938382)