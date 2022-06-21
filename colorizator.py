import torch
from torchvision.transforms import ToTensor
import numpy as np
import os
from networks.models import Colorizer
from denoising.denoiser import FFDNetDenoiser
from utils.utils import resize_pad
from ptflops import get_model_complexity_info
import cv2
import matplotlib.pyplot as plt
import math
from networks.RRDBNet import RRDBNet

# import coremltools as ct

class MangaColorizator:
    def __init__(self, device, generator_path, extractor_path,surperpath, superr, color_tile, sr_tile, tile_pad):
        self.superr = superr
        self.color_tile_size = color_tile
        self.sr_tile_size = sr_tile
        self.tile_pad = tile_pad
        self.surper_path = surperpath
        self.colorizer = Colorizer().to(device)
        m=torch.load(generator_path, map_location = device)
        #self.colorizer.generator.load_state_dict(m)
        self.colorizer.generator=m
        self.colorizer = self.colorizer.eval()
        self.model = self.colorizer
        srmodel = torch.load(self.surper_path, map_location=torch.device('cpu'))

        srmodel.eval()
        # torch.save(srmodel,"RealESRGAN_x4plus_anime_6B.pt",_use_new_zipfile_serialization=False)
        self.srmodel = srmodel.to(device)
        # if self.half:
        #     self.srmodel = self.srmodel.half()      
          
        # import torchvision

        # # Load a pre-trained version of MobileNetV2
        # torch_model = self.colorizer
        # # Set the model in evaluation mode.
        # torch_model.eval()

        # # Trace the model with random data.
        # example_input = torch.rand(1,5, 224, 224) 
        # traced_model = torch.jit.trace(torch_model.cpu(), example_input)
        # out = traced_model(example_input)

        # # Using image_input in the inputs parameter:
        # # Convert to Core ML program using the Unified Conversion API.
        # model = ct.convert(
        #     traced_model,
        #     convert_to="mlprogram",
        #     inputs=[ct.TensorType(shape=example_input.shape)],
        # )
        # # Save the converted model.
        # model.save("newmodel.mlpackage")        
        self.denoiser = FFDNetDenoiser(device)
        
        self.current_image = None
        self.current_hint = None
        self.current_pad = None
        
        self.device = device
        
    def set_image(self, image, size = 576, apply_denoise = True, denoise_sigma = 25, transform = ToTensor()):
        if (size % 32 != 0):
            raise RuntimeError("size is not divisible by 32")

        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma = denoise_sigma)
        #im=image[:, :, :1]
        #im=im.reshape((1,1200, 779))
        image, self.current_pad = resize_pad(image, size)
        self.current_image = transform(image).unsqueeze(0).to(self.device)
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2], self.current_image.shape[3]).float().to(self.device)
    
    def update_hint(self, hint, mask):
        '''
        Args:
           hint: numpy.ndarray with shape (self.current_image.shape[2], self.current_image.shape[3], 3)
           mask: numpy.ndarray with shape (self.current_image.shape[2], self.current_image.shape[3])
        '''
        
        if issubclass(hint.dtype.type, np.integer):
            hint = hint.astype('float32') / 255
            
        hint = (hint - 0.5) / 0.5
        hint = torch.FloatTensor(hint).permute(2, 0, 1)
        mask = torch.FloatTensor(np.expand_dims(mask, 0))

        self.current_hint = torch.cat([hint * mask, mask], 0).unsqueeze(0).to(self.device)

    def colorize(self):
        with torch.no_grad():
            # h=torch.cat([self.current_image, self.current_hint], 1)
            # macs, params = get_model_complexity_info(self.colorizer, (5,896,576), as_strings=True,print_per_layer_stat=True, verbose=True)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))		
            self.img=torch.cat([self.current_image, self.current_hint], 1)
            #self.tile_size=0

            #tile for color
            #tile for sr

            if self.color_tile_size > 0:
                fake_color1=self.tile_process(self.img, color_or_sr="color")
            else:
                fake_color1, _ = self.colorizer(self.img)
            color_result = fake_color1[0].detach().permute(1, 2, 0) * 0.5 + 0.5
            if self.current_pad[0] != 0:
                color_result = color_result[:-self.current_pad[0]]
            if self.current_pad[1] != 0:
                color_result = color_result[:, :-self.current_pad[1]]    
            # plt.imsave("1.png", color_result.cpu().numpy())   
            # color_img=cv2.imread("1.png")
            # img = torch.from_numpy(np.transpose(color_img, (2, 0, 1))).float()
            # color_result = img.unsqueeze(0).cuda()

            if self.superr:
                color_result=color_result.permute(2, 0, 1)
                color_result=color_result.unsqueeze(0)
                if self.sr_tile_size > 0:
                    srresult = self.tile_process(color_result.detach() , color_or_sr="sr")
                else:
                    srresult = self.srmodel(color_result.detach())
                output_img = srresult.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
                result = (output_img * 255.0).round().astype(np.uint8)
            else: 
                result = (color_result.detach().cpu().numpy()*255.0).round().astype(np.uint8)
            
        return result

    def tile_process(self,img,color_or_sr):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        if color_or_sr=="color":
            scale = 1
            tile_size=self.color_tile_size            
        elif color_or_sr=="sr":
            scale = 4#sr
            tile_size=self.sr_tile_size
        batch, channel, height, width = img.shape
        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, 3, output_height, output_width)

        # start with black image
        output = img.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        if color_or_sr=="color":
                            output_tile,_ = self.model(input_tile)

                        elif color_or_sr=="sr":
                            output_tile = self.srmodel(input_tile)                        
                except RuntimeError as error:
                    print('Error', error)
                if color_or_sr=="color":    
                    print(f'\tColor Tile {tile_idx}/{tiles_x * tiles_y}')
                elif color_or_sr=="sr":
                    print(f'\tSR Tile {tile_idx}/{tiles_x * tiles_y}')


                # output tile area on total image
                output_start_x = input_start_x * scale
                output_end_x = input_end_x  * scale
                output_start_y = input_start_y  * scale
                output_end_y = input_end_y  * scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad)  * scale
                output_end_x_tile = output_start_x_tile + input_tile_width  * scale
                output_start_y_tile = (input_start_y - input_start_y_pad)  * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
        return output
