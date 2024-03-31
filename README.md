# CodeFormer_TensorRT 

CodeFormer is a fantastic tech restoration with Codebook Lookup Transformer, check here for details, https://github.com/sczhou/CodeFormer/tree/master. Thanks very much for original author(s) for their great work. In this section, we base on this implementation, to optimize its inference performance on NVIDIA GPUs.

## System setup

Please check original system setup information from here, https://github.com/sczhou/CodeFormer/tree/master. Here shows my script just for reference:
>
> cd /thor/projects/codeformer/CodeFormer/
>
> python --version
>
> pip3 install -r requirements.txt
>
> python basicsr/setup.py develop
>
> pip install dlib
>
> python scripts/download_pretrained_models.py facelib
>
> python scripts/download_pretrained_models.py dlib
>
> python scripts/download_pretrained_models.py CodeFormer
>
> CUDA_VISIBLE_DEVICES=0 /usr/local/bin/nsys profile /usr/bin/python inference_codeformer.py -w 0.5 --has_aligned --input_path ./inputs/cropped_faces
>

## Performance optimizing

My main optimization scheme is to converting torch models to TensorRT and multi-thread overlapping execution of CPU/GPU. Finally, we get a speed up of 2.5X.
>
> 1. Baseline analysis
>
> 2. TRT inferencing
>
> 3. Move pre-processing to another thread
>
> 4. Move post-processing to another thread
>

Check the analysis details in the pdf document and we provide source code for each step too, check these files for detail:
>
> 1. inference_codeformer.py
>
> 2. inference_codeformer_1_TRT_inference_bs1_fp32.py
>
> 3. inference_codeformer_3_multi-thread-0.py
>
> 4. inference_codeformer_3_multi-thread-2.py
>
> 5. inference_codeformer_3_multi-thread-3.py
>
