<div align="center">
<img src="assets/logo.png" width="200">
<h3><span style="color:blue">L</span>ook back the <span style="color:blue">H</span>istory and <span style="color:blue">P</span>lan for the <span style="color:blue">F</span>uture in Autonomous Driving</h3>

[Sheng Wang](https://chantsss.github.io/)<sup>1</sup>, Yao Tian<sup>1</sup>, Xiaodong Mei<sup>1</sup>, Ge Sun<sup>1</sup>, Jie Cheng<sup>1</sup>, Fulong Ma<sup>2</sup>, [Pedro V. Sander](https://www.cse.ust.hk/~psander/)<sup>1</sup>, [Junwei Liang†](https://junweiliang.me/)<sup>1,2,†</sup>
†Corresponding authors
 
<sup>1</sup> HKUST, <sup>2</sup> HKUST(GZ)

<a href="https://chantsss.github.io/LHPF/">
<img src="https://img.shields.io/badge/Project-Page-blue?style=flat">
</a>
<a href='https://arxiv.org/abs/2411.17253' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>

![](assets/overview.png)

## Demo
<div style="display: flex; justify-content: space-between;">
    <video width="48%" alt="mp1_1" controls autoplay playsInline muted loop>
        <source src="assets/baseline.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video width="48%" alt="mp2_1" controls autoplay playsInline muted loop>
        <source src="assets/lhpf.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

<script type="text/javascript">
  document.addEventListener('DOMContentLoaded', function() {
      var videos = document.querySelectorAll('video');
      videos.forEach(function(video) {
          video.autoplay = true;
          video.controls = true;
          video.muted = true;
          video.loop = true;
          video.playsInline = true;
          video.playbackRate = 2.0;
          video.addEventListener('loadedmetadata', function() {
              this.playbackRate = 2.0;
          });
      });
  });
</script>

## News
**`05.12.2024`:** Github repo initialized.
**`27.11.2024`:** Project page released.
**`26.11.2024`:** Paper released on arXiv.
This work is under review, please stay tuned!


## Citation

If you find this repo useful, please consider giving us a star 🌟 and citing our related paper.

```bibtex
@misc{wang2024lhpflookhistoryplan,
      title={LHPF: Look back the History and Plan for the Future in Autonomous Driving}, 
      author={Sheng Wang and Yao Tian and Xiaodong Mei and Ge Sun and Jie Cheng and Fulong Ma and Pedro V. Sander and Junwei Liang},
      year={2024},
      eprint={2411.17253},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.17253}, 
}
```

