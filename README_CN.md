<p align="center">
  <img src="assets/logo.webp" height=200>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
An amazing manga colorization project  |  漫画AI上色
如果 Manga-Colorization 有帮助，请帮忙 ⭐ 这个 repo 或推荐给你的朋友😊 <br>

# 新的功能

- [x] 自动跳过彩色图片 将彩色图片复制到到输出文件夹。
- [x] 将原版提供的“zipfile”权重替换为旧的“pt”格式以支持 pytorch 版本 >=1.0
- [x] 为小显存 GPU 添加图片分块 选项。
- [x] 添加超分辨率 Real-ESRGAN（支持 分块）默认输出75 webp减少体积。

# 自动着色

1.下载 （generator.pt）(https://cubeatic.com/index.php/s/PcB4WgBnHXEKJrE)。将 'generator.pt' 放在 `./networks/` 中。
```bash
wget https://cubeatic.com/index.php/s/PcB4WgBnHXEKJrE/download -O ./networks/generator.pt
```
2.将图片放入“./input/”
3. 要为图片或图片文件夹上色，请使用以下命令：

使用 CPU：
```
$ python inference.py
```
使用 GPU：
```
$ python inference.py -g
```
4.彩色图像保存到“./output/”

---

####其它options

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

# 结果示例

|原图 |上色 |
|------------|-------------|
| <img src="input/0084.jpg" width="512"> | <img src="input/0083.jpg" width="512"> |
| <img src="output/0084.webp" width="512"> | <img src="output/0083.webp" width="512"> |
| <img src="input/017.jpg" width="512"> | <img src="input/016.jpg" width="512"> |
| <img src="output/017.webp" width="512"> | <img src="output/016.webp" width="512"> |
| <img src="input/bw2.jpg" width="512"> | <img src="output/bw2.webp" width="512"> |
| <img src="input/bw5.jpg" width="512"> | <img src="output/bw5.webp" width="512"> |

# 🤗 致谢

基于 https://github.com/qweasdd/manga-colorization-v2
感谢 https://github.com/xinntao/Real-ESRGAN
