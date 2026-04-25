# BayesOPTR - VAE

## Directory Layout
.
├── bayesopt
├── omar_dataset
├── simulation_data
│   ├── data
│   │   ├── output_good_4
│   │   ├── output_good_5
│   │   └── output_good_6
│   ├── extras
│   ├── output
│   ├── pipeline
│   ├── transoptr_verification
│   ├── x_centroid
│   └── y_centroid
├── transoptr
└── vae_project
    ├── experiments
    ├── tests
    └── vae

* bayesopt -- BOIS code with surrogate mean implementation and experiments
* omar_dataset -- real data collected from BOIS runs by Omar Hassan
* simulation_data -- data generation using randomly sampled misalignments in TRANSOPTR
* transoptr -- transoptr input file development
* vae_project -- vae and related experiments


## Requirements

* accpy: https://gitlab.triumf.ca/hla/acc-utilities/accpy/-/tree/alex?ref_type=heads
* pyoptr: https://gitlab.triumf.ca/beamphys/pyoptr/-/tree/alex?ref_type=heads 
* xml2optr:https://gitlab.triumf.ca/hla/acc-utilities/xml2optr/-/tree/master/xml2optr?ref_type=heads
* transoptr: https://gitlab.triumf.ca/beamphys/transoptr/-/tree/many-outputs?ref_type=heads

be careful to use the branches I've attached here. I've modified accpy, pyoptr, and transoptr to do this project. 


