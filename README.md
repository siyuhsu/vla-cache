# VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation
Official implementation of paper "[VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation](https://arxiv.org/abs/2502.02175)" (VLA-Cache), Under Review.  
By Siyu Xu, Yunke Wang, Chenghao Xia, Dihao Zhu, Tao Huang, Chang Xu.

:fire: **VLA-Cache: A training-free and plug-and-play approach to VLA acceleration.**


## Updates
* 10 Mar, 2025: We are planing to release the code.

## Overview
<details>

### Abstract

Vision-Language-Action (VLA) model can process instructions and visual perception to directly generate actions as output in an end-to-end fashion due to its strong multi-modal reasoning capabilities. While the performance of VLA models is promising, their computational cost can be substantial. This raises challenge for applying them on robotics tasks, which requires real-time decision-making to respond quickly to environmental changes. Since robotic control involves sequential decision-making, the visual input often exhibits minimal variation between successive steps. A natural idea is to reuse the computational results of unchanged visual tokens from the last step. Motivated by this idea, we propose VLA-Cache, an efficient vision-language-action model. VLA-Cache incorporates a token-selection mechanism that compares the visual input at each step with the input from the previous step, adaptively identifying visual tokens with minimal changes. The computational results for these unchanged tokens are then reused in subsequent steps via KV-cache, thereby significantly improving the efficiency of the VLA-Cache model. Experimental results on both simulation (e.g., LIBERO benchmark and SIMPLER) and real-world robot valid VLA-Cache can achieve practical acceleration with minimal sacrifice in success rate.


<p align='center'>
<img src='./assests/method.png' alt='mask' width='1200px'>
</p>

</details>

## License  
This project is released under the [Apache 2.0 license](LICENSE).

## Citation  
```
@article{xu2025vla,
  title={VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation},
  author={Xu, Siyu and Wang, Yunke and Xia, Chenghao and Zhu, Dihao and Huang, Tao and Xu, Chang},
  journal={arXiv preprint arXiv:2502.02175},
  year={2025}
}
```