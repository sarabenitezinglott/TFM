<h1 style="text-align: center;"> Approach 2: AI-Based Detection of Yeast Cell Cycle Stages </h1>
<h3 style="text-align: center;"> Oxidative Stress and Cell Cycle Research Group - PRBB </h3>
<h4 style="text-align: center;"> Sara del Carmen Benítez-Inglott González </h4>


## Introduction 

The duration of the cell cycle varies across different species. In the case of _Schizosaccharomyces pombe_ fission yeast, its cell cycle spans 8 hours, making it a preferred choice in numerous research laboratories. This preference stems not only from its distinctive cell cycle length but also from other resemblances it shares with human cells, such us gene structure, chromatin dynamics, prevalence of introns and the gene expression through pre-mRNA splicing, RNAi pathways and epigenetic gene silencing [[1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8193909/#:~:text=The%20fission%20yeast%20Schizosaccharomyces%20pombe,budding%20yeast%20Saccharomyces%20cerevisiae%2C%20S.)]. Furthermore, it is crucial for us to ascertain the duration of its cell cycle and to accurately detect the specific stage at which each yeast cell is positioned. In conclusion, this _Schizosaccharomyces pombe_ yeast is well known for its contribution as a model organism for the understanding of regulation and conservation of the eukariotic cell cycle.

For this apprach of identifying yeast cell cycle stages in a video, an GUI-Based Python framework called Cell-ACDC or Cell-Analysis of the Cell Division Cycle, is used. Cell-ACDC is an application which combines the best available neural network models. 


<p align="center">

|     tools        |Cell-ACDC|YeaZ|Cellpose|YeastMate|DeepCell|PhyloCell|CellProfiler|ImageJ/Fiji|YeastSpotter|YeastNet|MorphoLibJ|
|:----------------:|:-------:|:--:|:------:|:-------:|:------:|:-------:|:----------:|:---------:|:----------:|:------:|:--------:|
| Deep-Learning Segmentation|Y|Y|Y|Y|Y|N|Y|Y |Y |Y|N|
| Traditional Segmentation|Y|N|N|N|N|Y|Y|Y |N |N|Y|
| Tracking|Y|Y|N|N|Y|Y|Y|Y|N|N|N|
|Manual corrections|Y|Y|Y|Y|Y|Y|Y|Y |N|N|Y|
|Automatick handling of real-time traking|Y|N|N|N |N|N|N|N|N|N|N|
|Automatic propagation of correction|Y|N|N|N|N|Y|N|N|N|N| N|
|Automatic mother-bud pairing|Y|N|N|Y|N|Y|N|N|N|N|N|
|Pedigree Annotation|Y|N|N|Y|Y|Y|Y|Y|N|N|N|
|Cell Division Annotation|Y|N|N|N|N|Y|Y|Y|N|N|N|
|Downstream analysis|Y|N|N|N|Y|Y|Y|Y|N|N|N|
|Supports 3D z-stacks|Y|N|Y|N|Y|N|Y|Y|N|N|Y|
|Supports multiple model organism|Y|N|Y|N|Y|N|Y|Y|N|N|Y|
|Supports Bio-formats|Y|N|N|N|N|N|Y|Y|N |N|Y|
|User manual|Y|Y|Y|Y|Y|N|Y| Y|Y |Y|Y|
|Open Source|Y|Y|Y|Y|Y|Y |Y| Y|Y |Y|Y|
|Not licence requiered|Y| Y |Y|Y |Y | N |Y| Y|Y |Y|Y|

</p>

Cell-ACDC automatically computes several single-cell numerical features such as cell area and cell volume, plus the mean, max, median, sum and quantiles of any additional fluorescent channel's signal. It even performs background correction, to compute the protein amount and concentration.

## Aims

Identify the cell stage of each yeast in the video and tally the number of cell divisions from the initial ones.

## Challenges Faced 

This procedure follows different steps:

0. Create data structure from microscopy/image file(s).
1. Launch data prep module.
2. Launch segmentation module.
3. Launch GUI.

## Pipeline 

## Results

## Conclusions 

## Bibliography 

1. Vyas A, Freitas AV, Ralston ZA, Tang Z. Fission Yeast Schizosaccharomyces pombe: A Unicellular "Micromammal" Model Organism. Curr Protoc. 2021 Jun;1(6):e151. doi: 10.1002/cpz1.151. Erratum in: Curr Protoc. 2021 Jul;1(7):e225. PMID: 34101381; PMCID: PMC8193909.

## Links 

[Padovani, F., Mairhörmann, B., Schmoller, K., Lengefeld, J., & Falter-Braun, P. Cell-ACDC: segmentation, tracking, annotation and quantification of microscopy imaging data [Computer software]](https://github.com/SchmollerLab/Cell_ACDC)

[Padovani, F., Mairhörmann, B., Falter-Braun, P. et al. Segmentation, tracking and cell cycle analysis of live-cell imaging data with Cell-ACDC. BMC Biol 20, 174 (2022). https://doi.org/10.1186/s12915-022-01372-6](https://doi.org/10.1186/s12915-022-01372-6)

[Helmholtz Munich](https://doi.org/10.1186/s12915-022-01372-6)