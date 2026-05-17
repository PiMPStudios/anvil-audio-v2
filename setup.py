from setuptools import setup, find_packages

setup(
    name='anvil-audio',
    version='1.0.0',
    url='https://github.com/DaRealDaHoodie/anvil-audio',
    author='Dahoodie (fork of friendly-stable-audio-tools by Yukara Ikemiya and Stability AI)',
    description='A pluggable studio tool for AI audio generation. Swap models, keep your workflow.',
    python_requires='>=3.12',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'anvil=anvil_audio.cli:main',
        ],
    },
    install_requires=[
        'alias-free-torch>=0.0.6',
        'auraloss>=0.4.0',
        'descript-audio-codec==1.0.0',   # API-sensitive, keep pinned
        'einops>=0.6.0',
        'einops-exts>=0.0.4',
        'ema-pytorch>=0.2.3',
        'encodec>=0.1.1',
        'gradio>=3.42.0',
        'huggingface_hub>=0.19.0',
        'importlib-resources>=5.12.0',
        'k-diffusion==0.1.1',             # API-sensitive, keep pinned
        'laion-clap>=1.1.4',
        'local-attention>=1.8.6',
        'mcp>=1.9.0',
        'pandas>=2.0.0',
        'pedalboard>=0.7.4',
        'prefigure>=0.0.9',
        'pytorch_lightning>=2.1.0,<3.0.0',
        'PyWavelets>=1.4.0',
        'safetensors>=0.3.0',
        'sentencepiece>=0.1.99',
        's3fs',
        'torch>=2.0.1',
        'torchaudio>=2.0.2',
        'torchmetrics>=0.11.4',
        'tqdm',
        'transformers>=4.35.0',
        'v-diffusion-pytorch==0.0.2',     # API-sensitive, keep pinned
        'vector-quantize-pytorch>=1.9.14',
        'wandb>=0.15.4',
        'webdataset>=0.2.48',
        'x-transformers<1.27.0'           # upper-bound required by model code
    ],
    extras_require={
        'mlx': ['mlx-audiogen'],
        'intelligence': ['mlx-lm'],
        'dataset': ['yt-dlp>=2024.5.27'],
        'acestep': [
            'ace-step @ git+https://github.com/ace-step/ACE-Step-1.5.git',
        ],
        'apple': [
            'mlx-audiogen',
            'mlx-lm',
            'yt-dlp>=2024.5.27',
            'ace-step @ git+https://github.com/ace-step/ACE-Step-1.5.git',
        ],
        'cuda': [
            'ace-step @ git+https://github.com/ace-step/ACE-Step-1.5.git',
        ],
    },
)
