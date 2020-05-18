# DAVIS_Challenge2020_interactive_segment
CVPR Workshop, DAVIS Challenge 2020, Interactive Segmentation, US+UIT, VNU-HCM

# Requirements
CUDA=9.0 <br>
GPU Usage: 11GB

# Setup
  1. Install CUDA=9.0 <br>
      <pre>
      export PATH=/usr/local/cuda-9.0/bin:${PATH}
      export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}
      export CUDA_HOME=/usr/local/cuda-9.0 </pre>
  2. Install conda environment (refer requirements.pip.txt or requirements.conda.txt)
  3. Modify absolute paths for data dir, weight files (try running and then fix)

# Usage
  <pre>
  cd interactive
  python submit_chiet.py </pre>

# References
  1. SiamMask: https://github.com/foolwood/SiamMask
  <pre>
    @inproceedings{wang2019fast,
        title={Fast online object tracking and segmentation: A unifying approach},
        author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
        booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
        year={2019}
    } </pre>
  2. FBRS Interactive Segmentation: https://github.com/saic-vul/fbrs_interactive_segmentation
  <pre>
      @article{fbrs2020,
        title={f-BRS: Rethinking Backpropagating Refinement for Interactive Segmentation},
        author={Konstantin Sofiiuk, Ilia Petrov, Olga Barinova, Anton Konushin},
        journal={arXiv preprint arXiv:2001.10331},
        year={2020}
    } </pre>
  3. CascadePSP: https://github.com/hkchengrex/CascadePSP
  <pre>
    @inproceedings{CascadePSP2020,
      title={CascadePSP: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement},
      author={Cheng, Ho Kei and Chung, Jihoon and Tai, Yu-Wing and Tang, Chi-Keung},
      booktitle={CVPR},
      year={2020}
  } </pre>
