import os, sys, math, random
from torchvision.utils import save_image
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset
from utils.util_image import ImageSpliterTh

class BaseSampler:
    def __init__(
            self,
            configs,
            sf=None,
            use_fp16=False,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            desired_min_size=64,
            seed=10000,
            ddim=False
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_fp16 = use_fp16
        self.desired_min_size = desired_min_size
        self.ddim=ddim
        if sf is None:
            sf = configs.diffusion.params.sf
        self.sf = sf

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        # assert num_gpus == 1, 'Please assign one available GPU using CUDA_VISIBLE_DEVICES!'

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        if self.use_fp16:
            model.dtype = torch.float16
            model.convert_to_fp16()
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half()
            else:
                self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

class Sampler(BaseSampler):    
    def sample_func(self, y0, one_step=False, apply_decoder=True):
        '''
        Input:
            y0: Ảnh chất lượng thấp, dạng tensor [n,c,h,w], giá trị nằm trong khoảng [−1,1], RGB
            one_step: Nếu True (SinSR), mô hình thực hiện single-step diffusion (chỉ một bước)
        Output:
            sample: Ảnh siêu phân giải đầu ra, dạng tensor [n,c,h,w], giá trị nằm trong khoảng [−1,1], RGB
        '''

        '''
        Đảm bảo kích thước của ảnh y0 chia hết cho desired_min_size (kích thước tối thiểu yêu cầu).
        Padding: Nếu ảnh không thỏa mãn điều kiện, padding được thêm vào 
        (với phương pháp phản xạ, reflect) để mở rộng kích thước ảnh.
        '''
        desired_min_size = self.desired_min_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False


        model_kwargs={'lq':y0,} if self.configs.model.params.cond_lq else None
        
        # p_sample_loop: Đây là quá trình khuếch tán đa bước (multi-step diffusion)
        # xT =>xT−1 => x0 (giảm nhiễu qua nhiều bước).
        # Nếu one_step=True, quá trình này chỉ thực hiện một bước duy nhất.
        results = self.base_diffusion.p_sample_loop(
                y=y0,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=False,
                clip_denoised=(self.autoencoder is None),
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                one_step=one_step,
                apply_decoder=apply_decoder
                )
        
        if flag_pad and apply_decoder:
            # Sau khi quá trình suy luận hoàn tất, nếu ảnh được padding trước đó, phần thừa sẽ được cắt bỏ.
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]
            
        if not apply_decoder:
            # Nếu không giải mã ảnh, trả về kết quả trong không gian tiềm ẩn (pred_xstart).
            return results["pred_xstart"]
        
        # Giải mã
        return results.clamp_(-1.0, 1.0)
    
    def inference(self, in_path, out_path, one_step=False, return_tensor=False, apply_decoder=True):
        '''
        suy luận (inference), áp dụng mô hình SinSR để xử lý ảnh suy giảm (low-quality image, LQ) 
        và tạo ra ảnh siêu phân giải (super-resolution image, SR).
        
        Input:
            in_path: Đường dẫn đến ảnh đầu vào (ảnh suy giảm) 
            out_path: Đường dẫn thư mục lưu kết quả đầu ra (ảnh siêu phân giải).
        '''
        def _process_per_image(im_lq_tensor):
            '''
            Ap dụng mô hình đã được huấn luyện lên một ảnh suy giảm để tạo ra ảnh siêu phân giải.

            Input:
                im_lq_tensor: Tensor PyTorch với kích thước [b, c, h, w], RGB
                    b: Batch size (số lượng ảnh trong batch, trường hợp này là 1).
                    c: Số lượng kênh (3 đối với ảnh RGB).
                    h: Chiều cao của ảnh.
                    w: Chiều rộng của ảnh.
                Dải giá trị: [0, 1].

            Output:
                im_sr: Tensor PyTorch, [1, c, h, w] (1 ảnh siêu phân giải), RGB
                Dải giá trị: [0, 1].
            '''

            # Kiểm tra xem chiều cao (h) hoặc chiều rộng (w) của ảnh có lớn hơn chop_size không
            # Nếu ảnh lớn hơn, nó sẽ được chia thành các phần nhỏ để xử lý. 
            # Điều này đảm bảo rằng ảnh có thể được xử lý trong bộ nhớ GPU hạn chế.
            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.sf,
                        extra_bs=self.chop_bs,  
                        )
                
                # Mỗi phần nhỏ im_lq_pch được xử lý riêng lẻ thông qua hàm self.sample_func.
                for im_lq_pch, index_infos in im_spliter:
                    # Thực hiện suy luận (inference) với mô hình khuếch tán (diffusion model).
                    # Dự đoán ảnh siêu phân giải từ ảnh suy giảm.
                    # kết quả: im_sr_pch => Tensor siêu phân giải 
                    im_sr_pch = self.sample_func(
                            (im_lq_pch - 0.5) / 0.5, # Tiền xử lý: Chuẩn hóa tensor từ [0, 1] sang [-1, 1].
                            one_step=one_step, apply_decoder=apply_decoder
                            )     # 1 x c x h x w, [-1, 1]
                    # Lưu trữ kết quả của từng phần nhỏ.
                    im_spliter.update(im_sr_pch.detach(), index_infos)
                
                # Kết hợp các phần nhỏ thành ảnh hoàn chỉnh:
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                        (im_lq_tensor - 0.5) / 0.5,
                        one_step=one_step, apply_decoder=apply_decoder
                        )     # 1 x c x h x w, [-1, 1]

            if apply_decoder:
                # Chuyển giá trị từ dải [-1, 1] về [0, 1].
                im_sr_tensor = im_sr_tensor * 0.5 + 0.5

            return im_sr_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
        if not out_path.exists():
            out_path.mkdir(parents=True)
        
        return_res = {}
        if not in_path.is_dir():
            # Đọc ảnh low quality từ in_path, chuyển ảnh sang dạng ma trận kích thước [h, w, c] (chiều cao, chiều rộng, số kênh màu).
            im_lq = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
            # Chuyển ảnh low quality sang tensor
            im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w

            # Trả về tensor đầu ra là ảnh super-resolution với kích thước [1, c, h, w]
            im_sr_tensor = _process_per_image(im_lq_tensor)
            # Chuyển tensor SR về ảnh
            im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))

            im_path = out_path / f"{in_path.stem}.png"
            util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
            if return_tensor:
                return_res[im_path.stem]=im_sr_tensor
        
        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")
        return return_res
    
if __name__ == '__main__':
    pass

