<p align="center">
  <img src="assets/logo.webp" height=200>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
An amazing manga colorization project  |  漫画AI上色

# New features

- [x] Skip color img just copy to output.
- [x] Replace the offered "zipfile" weight to old "pt" format to support pytorch version >=1.0
- [x] Add tile img option for small cuda memory gpu.
- [x] Add Real-ESRGAN (support tile) for output super-resolution.

# Automatic colorization

1. Download [generator](https://cubeatic.com/index.php/s/PcB4WgBnHXEKJrE). Put 'generator.pt' in `./networks/`
```bash
wget https://cubeatic.com/index.php/s/PcB4WgBnHXEKJrE/download -O ./networks/generator.pt
```
2. Put imgs into "./input/"
3. To colorize image or folder of images, use the following command:

USE CPU:
```
$ python inference.py
```
USE GPU:
```
$ python inference.py -g
```
4. Colorized image saved to "./output/"

---

If Manga-Colorization is helpful, please help to ⭐ this repo or recommend it to your friends 😊 <br>

# Usage of python script

```
usage: inference.py [-h] [-p PATH] [-op OUTPUTPATH] [-gen GENERATOR]
                    [-sur SURPERPATH] [-ext EXTRACTOR] [-g] [-nd]
                    [-ds DENOISER_SIGMA] [-s SIZE] [-ct COLORTILE]
                    [-st SRTILE] [--tile_pad TILE_PAD] [-sr]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  input dir/file
  -op OUTPUTPATH, --outputpath OUTPUTPATH
                        output dir
  -gen GENERATOR, --generator GENERATOR
  -sur SURPERPATH, --surperpath SURPERPATH
  -ext EXTRACTOR, --extractor EXTRACTOR
  -g, --gpu             Use gpu
  -nd, --no_denoise     No denoiser before color
  -ds DENOISER_SIGMA, --denoiser_sigma DENOISER_SIGMA
                        Denoiser_sigma
  -s SIZE, --size SIZE  Color output size
  -ct COLORTILE, --colortile COLORTILE
                        Color Tile size, 0 for no tile
  -st SRTILE, --srtile SRTILE
                        SR Tile size, 0 for no tile
  --tile_pad TILE_PAD   Tile padding
  -sr, --superr         SR or not SR by RealESRGAN_x4plus_anime_6B
                        aftercolored
```

# Samples

| Original      | Colorization      |
|------------|-------------|
| <img src="input/0084.jpg" width="512"> | <img src="input/0083.jpg" width="512"> |
| <img src="output/0084.webp" width="512"> | <img src="output/0083.webp" width="512"> |
| <img src="input/017.jpg" width="512"> | <img src="input/016.jpg" width="512"> |
| <img src="output/017.webp" width="512"> | <img src="output/016.webp" width="512"> |
| <img src="input/bw2.jpg" width="512"> | <img src="output/bw2.webp" width="512"> |
| <img src="input/bw5.jpg" width="512"> | <img src="output/bw5.webp" width="512"> |

## 🤗 Acknowledgement

Based on https://github.com/qweasdd/manga-colorization-v2
Thx https://github.com/xinntao/Real-ESRGAN
