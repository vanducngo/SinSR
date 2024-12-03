import enum
import math
from torchvision.utils import save_image
import numpy as np
import torch as th
import torch.nn.functional as F

from .basic_ops import mean_flat
from .losses import normal_kl

# from ldm.models.autoencoder import AutoencoderKLTorch

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return np.linspace(
            beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
        )**2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        kwargs=None):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        # ponential = kwargs.get('ponential', None)
        # start = math.exp(math.log(min_noise_level / kappa) / ponential)
        # end = math.exp(math.log(etas_end) / (2*ponential))
        # xx = np.linspace(start, end, num_diffusion_timesteps, endpoint=True, dtype=np.float64)
        # sqrt_etas = xx**ponential
        power = kwargs.get('power', None)
        etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        increaser = math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    elif schedule_name == 'ldm':
        import scipy.io as sio
        mat_path = kwargs.get('mat_path', None)
        sqrt_etas = sio.loadmat(mat_path)['sqrt_etas'].reshape(-1)
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon
    PREVIOUS_X = enum.auto()  # the model predicts epsilon
    RESIDUAL = enum.auto()  # the model predicts epsilon
    EPSILON_SCALE = enum.auto()  # the model predicts epsilon

class LossType(enum.Enum):
    MSE = enum.auto()           # simplied MSE
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL

class ModelVarTypeDDPM(enum.Enum):
    """
    What is used as the model's output variance.
    """

    LEARNED = enum.auto()
    LEARNED_RANGE = enum.auto()
    FIXED_LARGE = enum.auto()
    FIXED_SMALL = enum.auto()

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param sqrt_etas: a 1-D numpy array of etas for each diffusion timestep,
                starting at T and going to 1.
    :param kappa: a scaler controling the variance of the diffusion kernel
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param loss_type: a LossType determining the loss function to use.
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param scale_factor: a scaler to scale the latent code
    :param sf: super resolution factor
    """

    def __init__(
        self,
        *,
        sqrt_etas,
        kappa,
        model_mean_type,
        loss_type,
        sf=4,
        scale_factor=None,
        normalize_input=True,
        latent_flag=True,
    ):
        self.kappa = kappa
        self.model_mean_type = model_mean_type
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.normalize_input = normalize_input
        self.latent_flag = latent_flag
        self.sf = sf

        # Use float64 for accuracy.
        self.sqrt_etas = sqrt_etas
        self.etas = sqrt_etas**2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        
        # coefficient for DDIM inference
        self.etas_prev_clipped = np.append(
                self.etas_prev[1], self.etas_prev[1:]
                )
        self.ddim_coef1 = self.etas_prev * self.etas
        self.ddim_coef2 = self.etas_prev / self.etas

        # weight for the mse loss
        if model_mean_type in [ModelMeanType.START_X, ModelMeanType.RESIDUAL]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2
        elif model_mean_type in [ModelMeanType.EPSILON, ModelMeanType.EPSILON_SCALE]  :
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (
                    kappa * self.alpha / ((1-self.etas) * self.sqrt_etas)
                    )**2
        else:
            raise NotImplementedError(model_mean_type)

        # self.weight_loss_mse = np.append(weight_loss_mse[1],  weight_loss_mse[1:])
        self.weight_loss_mse = weight_loss_mse

    def q_mean_variance(self, x_start, y, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_sample(self, x_start, y, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_start
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x_t, y, t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)
        model_output = model(self._scale_input(x_t, t), t, **model_kwargs)

        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        # ddim coef
        ddim_coef1 = _extract_into_tensor(self.ddim_coef1, t, x_t.shape) # etas_pre*etas
        ddim_coef2 = _extract_into_tensor(self.ddim_coef2, t, x_t.shape) # etas_pre/etas
        etas = _extract_into_tensor(self.etas, t, x_t.shape)
        etas_prev = _extract_into_tensor(self.etas_prev, t, x_t.shape)
        k = (1-etas_prev+th.sqrt(ddim_coef1)-th.sqrt(ddim_coef2))
        m = th.sqrt(ddim_coef2)
        j = (etas_prev - th.sqrt(ddim_coef1))
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_xstart = process_xstart(model_output)
        elif self.model_mean_type == ModelMeanType.RESIDUAL:      # predict x_0
            pred_xstart = process_xstart(
                self._predict_xstart_from_residual(y=y, residual=model_output)
                )
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps_scale(x_t=x_t, y=y, t=t, eps=model_output)
            )                                                  #  predict \eps
        else:
            raise ValueError(f'Unknown Mean type: {self.model_mean_type}')

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            # used for ddim
            "ddim_k": k,
            "ddim_m": m,
            "ddim_j": j,
            # "etas": etas,
            # "etas_prev": etas_prev
        }

    def _predict_xstart_from_eps(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - _extract_into_tensor(self.sqrt_etas, t, x_t.shape) * self.kappa * eps
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_eps_scale(self, x_t, y, t, eps):
        assert x_t.shape == eps.shape
        return  (
            x_t - eps - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(1 - self.etas, t, x_t.shape)

    def _predict_xstart_from_residual(self, y, residual):
        assert y.shape == residual.shape
        return (y - residual)

    def _predict_eps_from_xstart(self, x_t, y, t, pred_xstart):
        return (
            x_t - _extract_into_tensor(1 - self.etas, t, x_t.shape) * pred_xstart
                - _extract_into_tensor(self.etas, t, x_t.shape) * y
        ) / _extract_into_tensor(self.kappa * self.sqrt_etas, t, x_t.shape)
    
    def ddim_inverse(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t} from the model at the given timestep (x_{t-1}).

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            clip_denoised=False,
            denoised_fn=None,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        sample = (x - pred_xstart * out["ddim_k"] - out["ddim_j"] * y) / out["ddim_m"]
        return {"sample": sample, "pred_xstart": pred_xstart}


    def p_sample(self, model, x, y, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}



    def p_sample_loop(
        self,
        y,
        model,
        first_stage_model=None,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        one_step=False,
        apply_decoder=True
    ):
        """
        Tái tạo dữ liệu từ trạng thái nhiễu
        
        Đầu vào:
            y: Tensor đầu vào [N,C,H,W], low quality image
            model: Mô hình diffusion được sử dụng.
            first_stage_model: the autoencoder model
            noise: Nhiễu Gaussian được thêm vào dữ liệu đầu vào y.
            clip_denoised:  Nếu True, đầu ra x0 được giới hạn trong khoảng [−1,1]
            denoised_fn: Hàm tuỳ chỉnh áp dụng lên dự đoán x0
            model_kwargs: Tham số bổ sung truyền vào mô hình
            progress: Hiển thị thanh tiến trình (progress bar) nếu True
        
        Đầu ra:
            Một batch dữ liệu đã tái tạo
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            y,
            model,
            first_stage_model=first_stage_model,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            one_step=one_step
        ):
            final = sample
        if apply_decoder:
            return self.decode_first_stage(final["sample"], first_stage_model)
        return final

    def p_sample_loop_progressive(
            self, y, model,
            first_stage_model=None,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            one_step=False
    ):
        """
        Đầu vào:
            - Kế thừa từ hàm p_sample_loop()
        
        Đầu ra:
            - Trả về các trạng thái trung gian hoặc kết quả cuối cùng (x0).
        """
        
        # Dữ liệu đầu vào y được mã hóa vào không gian tiềm ẩn
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)

        # Tạo nhiễu Gaussian
        if noise is None:
            noise = th.randn_like(z_y)
        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        # indices: Danh sách các timestep t theo thứ tự ngược (từ T đến 0)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * y.shape[0], device=device)
            with th.no_grad():
                # Gọi hàm p_sample để dự đoán trạng thái tiếp theo x_t−1 từ xt
                out = self.p_sample(
                    model,
                    z_sample,
                    z_y,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                if one_step:
                    # bỏ qua các bước trung gian, trả về x0 ngay sau một bước
                    out["sample"]=out["pred_xstart"]
                    yield out
                    break
                yield out

                # Cập nhật trạng thái z_sample để sử dụng trong bước tiếp theo.
                z_sample = out["sample"]

    def decode_first_stage(self, z_sample, first_stage_model=None, no_grad=True):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        else:
            if no_grad:
                with th.no_grad():
                    z_sample = 1 / self.scale_factor * z_sample
                    z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                    out = first_stage_model.decode(z_sample)
            else:
                z_sample = 1 / self.scale_factor * z_sample
                z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                out = first_stage_model.decode(z_sample, grad_forward=True)
            return out.type(ori_dtype)
    
    """
    đầu ra
        + Nếu first_stage_model=None:
            Trả về dữ liệu đầu vào y mà không qua mã hóa.
        + Nếu first_stage_model được cung cấp:
            Trả về dữ liệu đã mã hóa (z_y), đã được điều chỉnh bởi scale_factor.
    """
    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            # Tăng kích thước lên theo tỷ lệ (scale_factor) bằng phương pháp nội suy bicubic.
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        
        if first_stage_model is None:
            return y
        else:
            # Xử lý mã hóa với mô hình autoencoder

            # th.no_grad(): Vô hiệu hóa tính toán gradient để tiết kiệm bộ nhớ
            with th.no_grad():
                # Điều chỉnh kiểu dữ liệu của y sao cho khớp với kiểu dữ liệu của các tham số trong first_stage_model.
                y = y.type(dtype=next(first_stage_model.parameters()).dtype)
                # Mã hóa dữ liệu y từ không gian đầu vào sang không gian tiềm ẩn, kết quả là z_y.
                z_y = first_stage_model.encode(y)
                # Điều chỉnh giá trị của dữ liệu mã hóa bằng cách nhân với scale_factor.
                out = z_y * self.scale_factor
                # Trả về dữ liệu đã mã hóa với kiểu dữ liệu gốc (ori_dtype).
                return out.type(ori_dtype)

    """
    Đầu vào: 
        + y: Dữ liệu suy giảm.
        + ϵ: Nhiễu Gaussian (tuỳ chọn, nếu không có sẽ được sinh tự động).
    
    Đầu ra: 
        + xT: Dữ liệu bị nhiễu mạnh nhất tại bước cuối cùng T

    Mục đích: Sinh dữ liệu đầu vào bị nhiễu mạnh để khởi tạo quá trình huấn luyện hoặc suy luận ngược (reverse diffusion).
    """
    def prior_sample(self, y, noise=None):
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """

        # Sinh nhiễu Gaussian nếu không được cung cấp
        if noise is None:
            noise = th.randn_like(y)

        # Thời điểm khuếch tán t được đặt ở bước cuối cùng của quá trình khuếch tán (𝑡 =𝑇)
        # self.num_timesteps - 1: Tổng số bước khuếch tán trừ 1, tương ứng với bước cuối cùng.
        t = th.tensor([self.num_timesteps-1,] * y.shape[0], device=y.device).long()

        # xT =y + κ*sqrt(ηs) * ϵ)
        #   y: Ảnh bị suy giảm (degraded input), được coi là trung tâm của phân phối prior.
        #   ϵ: Nhiễu Gaussian ngẫu nhiên (nếu không được cung cấp, sẽ được sinh ra).
        #   κ*sqrt(ηs): Hệ số điều chỉnh phương sai của nhiễu tại thời điểm t = T
        return y + _extract_into_tensor(self.kappa * self.sqrt_etas, t, y.shape) * noise

    """
    * Hàm training_losses_distill tính toán giá trị mất mát để:
            + Huấn luyện mô hình học sinh (student model):
                + Mô hình học sinh học cách tái tạo đầu ra giống với mô hình giáo viên.
                + Hoặc học cách dự đoán kết quả gần đúng với dữ liệu gốc (ground truth).
            + Truyền tri thức (Knowledge Distillation)
                + Truyền tri thức từ mô hình giáo viên sang mô hình học sinh.
                + Sử dụng phương pháp khuếch tán (diffusion) để dự đoán ảnh có độ phân giải cao.
        => Huấn luyện mô hình học sinh thông qua distillation, sao cho nó tái tạo được đầu ra chính xác nhất từ dữ liệu nhiễu hoặc suy giảm.
    
    * Step:
    -> Chuẩn bị dữ liệu: Xử lý ảnh đầu vào, thêm nhiễu, và chuyển đổi sang không gian tiềm ẩn.
    -> Lấy mục tiêu: Sử dụng đầu ra từ mô hình giáo viên hoặc ground truth.
    -> Tính toán mất mát: So sánh đầu ra của mô hình học sinh với mục tiêu.
    -> Hỗ trợ tùy chọn nâng cao: Hỗ trợ học xT, tinh chỉnh bằng ground truth


    * Đầu vào:
        + model: Mô hình học sinh đang được huấn luyện.
        + teacher_model: Mô hình giáo viên được sử dụng để sinh đầu ra làm mục tiêu (target).
        + x_start: Ảnh gốc
        + y: Ảnh bị suy giảm
        + t: Thời điểm khuếch tán (timesteps), kích thước [N].
        + first_stage_model: Mô hình autoencoder để mã hóa và giải mã không gian tiềm ẩn.
        + noise: Nhiễu Gaussian được thêm vào dữ liệu, kích thước giống z_y. Nếu không có, sẽ được tạo ngẫu nhiên.
        + learn_xT: Cho phép mô hình học cách dự đoán trạng thái nhiễu ban đầu xT.
        + finetune_use_gt: Dùng ground truth thay vì mô hình giáo viên để làm mục tiêu huấn luyện.

    * Đầu ra: 
        + term: Dictionary chứa các giá trị mất mát và các thành phần liên quan, ví dụ:
            + terms["loss"]: Tổng giá trị mất mát.
            + terms["loss_xT"]: Mất mát liên quan đến trạng thái nhiễu xT.
            + terms["loss_gt"]: Mất mát so với ground truth.
        + z_t: Dữ liệu đầu vào bị nhiễu tại thời điểm t, sau khi thêm noise.
        + pred_zstart: Dự đoán cuối cùng từ mô hình học sinh, có thể là:
            + z_start (dự đoán trực tiếp từ mô hình giáo viên hoặc ground truth).
            + xT (trạng thái ban đầu của nhiễu, nếu tùy chọn learn_xT được bật).
    """
    def training_losses_distill(
            self, model, teacher_model, x_start, y, t,
            first_stage_model=None,
            model_kwargs=None,
            noise=None, 
            learn_xT=False, 
            finetune_use_gt=False
            ):
        
        if model_kwargs is None:
            model_kwargs = {}
            
        # Tạo dữ liệu đầu vào cho quá trình khuếch tán (diffusion).
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True) # TODO can be eliminated to speed up, since z_y is already obtained in self.ddim_sample_loop/p_sample_loop
        
        if noise is None:
            noise = th.randn_like(z_y)
        
        terms = {}
        loss_type = "mse" # "mse"
        assert loss_type in ["mse", "mae"]
        terms["loss"] = 0
        
        # z_t: Dữ liệu đầu vào nhiễu, lấy mẫu từ quá trình khuếch tán.
        z_t = self.prior_sample(z_y, noise)
        
        # pred_zstart: Dự đoán cuối cùng từ mô hình học sinh.
        pred_zstart = None
        
        # Lấy đầu ra của mô hình giáo viên để làm mục tiêu huấn luyện.
        # sử dụng phương pháp DDIM (Denoising Diffusion Implicit Model)
        # Trả về một dự đoán của trạng thái đầu tiên (ảnh gốc hoặc phiên bản sạch) 
        # được mô hình giáo viên tái tạo từ dữ liệu đầu vào bị suy giảm y và nhiễu ϵ
        # => lấy mẫu ngược (reverse sampling) từ xT -> x0
        z_start_teacher = self.ddim_sample_loop(y, teacher_model, noise, first_stage_model, clip_denoised=True if first_stage_model is None else False, apply_decoder=False, model_kwargs=model_kwargs)["sample"]

        # LossType là MSE
        if self.loss_type == LossType.MSE:
            # Dự đoán đầu ra từ mô hình học sinh
            model_output = model(self._scale_input(z_t, t), t, **model_kwargs)
            
            z_start = z_start_teacher

            # Mục tiêu cho mô hình học sinh
            target = {
                ModelMeanType.START_X: z_start, # Đầu ra từ Mô hình giáo viên
                ModelMeanType.RESIDUAL: z_y - z_start, # Hiệu giữa đầu vào nhiễu và đầu ra.
                ModelMeanType.EPSILON: noise, # Nhiễu gốc.
                # EPSILON_SCALE: Nhiễu gốc được nhân với trọng số.
                ModelMeanType.EPSILON_SCALE: noise*self.kappa*_extract_into_tensor(self.sqrt_etas, t, noise.shape),
            }[self.model_mean_type]

            assert model_output.shape == target.shape   

            # Mất mát giữa đầu ra của mô hình học sinh và mục tiêu (target).
            # MSE
            terms[loss_type] = mean_flat((target - model_output) ** 2)
            
            if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                terms[loss_type] /= (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
            if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            else:
                weights = 1
            
            # Tổng giá trị mất mát.
            terms["loss"] += terms[loss_type] * weights
            
            # Config true from SinSR
            if learn_xT:
                # Mô hình học sinh sẽ học cách dự đoán trạng thái nhiễu xT (trạng thái bị nhiễu mạnh nhất)
                # z_start_teacher: Z-start được dự đoán từ mô hình giáo viên.
                predicted_xT = model(self._scale_input(z_start_teacher, t), t*0, **model_kwargs)
                terms[loss_type+"_xT"] = mean_flat((z_t - predicted_xT) ** 2) # MSE
                terms["loss"] += terms[loss_type+"_xT"]   
                    
        else:
            raise NotImplementedError(self.loss_type)
        
        # detach() tạo ra một tensor mới với cùng dữ liệu như model_output, 
        # nhưng không liên kết với đồ thị tính toán gradient.
        pred_zstart = model_output.detach()
               
        # Sử dụng ground truth để huấn luyện
        #  Mô hình học sinh sẽ học cách dự đoán trạng thái x0  hoặc 𝑧_start trực tiếp từ ground truth 𝑥_start, 
        # thay vì hoàn toàn dựa vào mô hình giáo viên.
        # Điều này giúp cải thiện khả năng tổng quát của mô hình học sinh bằng cách bổ sung thông tin từ dữ liệu thực.
        if finetune_use_gt:
            # Chuẩn bị dữ liệu ground truth để làm mục tiêu trực tiếp cho quá trình huấn luyện.
            z_start_gt=self.encode_first_stage(x_start, first_stage_model, up_sample=False)

            # th.no_grad(): Vô hiệu hóa tính toán gradient để tiết kiệm bộ nhớ
            with th.no_grad():
                # Dự đoán trạng thái nhiễu xT từ trạng thái gốc z_start_gt
                # t*0 => T = 0 => Trạng thái gốc.
                predicted_xT_from_gt = model(self._scale_input(z_start_gt, t), t*0, **model_kwargs)
            
            # predicted_xT_from_gt.detach(): Tách trạng thái dự đoán xT ra khỏi đồ thị gradient.
            # đảm bảo không làm ảnh hưởng đến gradient của các bước trước.
            # Đầu ra của mô hình học sinh, dự đoán trạng thái gốc zStart dựa trên Ground truth
            model_output_pedict_gt = model(self._scale_input(predicted_xT_from_gt.detach(), t), t, **model_kwargs)
            
            # Mất mát: So sánh giữa trạng thái gốc z_start_gt và đầu ra dự đoán model_output_pedict_gt
            terms[loss_type+"_gt"] = mean_flat((z_start_gt - model_output_pedict_gt) ** 2) #MSE
            terms["loss"] += (terms[loss_type+"_gt"]*finetune_use_gt)
            
            if pred_zstart is None: pred_zstart=model_output_pedict_gt
        
        #terms: Dictionary chứa các giá trị mất mát khác nhau (e.g., loss, loss_xT, loss_gt).
        # z_t: Đầu vào nhiễu.
        # pred_zstart: Dự đoán cuối cùng từ mô hình học sinh.
        return terms, z_t, pred_zstart
        
        
    def cov_loss(self, noise):
        feat = noise
        kernel_size=8
        b, c, h, w = feat.shape
        feat = feat.view(b*c, 1, h, w)

        feat_unfold = F.unfold(feat, kernel_size=kernel_size, stride=1)
        
        # n_patch = feat_unfold.shape[-1]
        # ratio = 0.1
        # feat_unfold = feat_unfold[..., th.randperm(n_patch)[...,:int(n_patch*ratio)]]
        
        feat_flatten = feat_unfold.permute(0,2,1).contiguous()
        def batch_cov(points):
            B, N, D = points.size()
            mean = points.mean(dim=1).unsqueeze(1)
            diffs = (points - mean).reshape(B * N, D)
            prods = th.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
            bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
            return bcov  # (B, D, D)
        cov = batch_cov(feat_flatten)
        target_cov =  th.eye(cov.shape[1]).repeat([cov.shape[0],1,1]).to(cov.device) * ((self.kappa * self.sqrt_etas)[-1])**2
        loss_cov = mean_flat((target_cov - cov)**2).view(b, c).sum(dim=1)
        return loss_cov

    def training_losses(
            self, model, x_start, y, t,
            first_stage_model=None,
            model_kwargs=None,
            noise=None,
            ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param first_stage_model: autoencoder model
        :param x_start: the [N x C x ...] tensor of inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
        z_start = self.encode_first_stage(x_start, first_stage_model, up_sample=False)

        if noise is None:
            noise = th.randn_like(z_start)

        z_t = self.q_sample(z_start, z_y, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.WEIGHTED_MSE:
            model_output = model(self._scale_input(z_t, t), t, **model_kwargs)
            target = {
                ModelMeanType.START_X: z_start,
                ModelMeanType.RESIDUAL: z_y - z_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.EPSILON_SCALE: noise*self.kappa*_extract_into_tensor(self.sqrt_etas, t, noise.shape),
            }[self.model_mean_type]
            assert model_output.shape == target.shape == z_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if self.model_mean_type == ModelMeanType.EPSILON_SCALE:
                terms["mse"] /= (self.kappa**2 * _extract_into_tensor(self.etas, t, t.shape))
            if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            else:
                weights = 1
            terms["loss"] = terms["mse"] * weights
        else:
            raise NotImplementedError(self.loss_type)

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, y=z_y, t=t, eps=model_output.detach())
        elif self.model_mean_type == ModelMeanType.RESIDUAL:
            pred_zstart = self._predict_xstart_from_residual(y=z_y, residual=model_output.detach())
        elif self.model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_zstart = self._predict_xstart_from_eps_scale(x_t=z_t, y=z_y, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                # the variance of latent code is around 1.0
                std = th.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm

    def ddim_sample(
        self,
        model,
        x,
        y,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        ddim_eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model=model,
            x_t=x,
            y=y,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        pred_xstart = out["pred_xstart"]
        
        # residual = y - pred_xstart
        # eps = self._predict_eps_from_xstart(x, y, t, pred_xstart)
        # etas = _extract_into_tensor(self.etas, t, x.shape)
        # etas_prev = _extract_into_tensor(self.etas_prev, t, x.shape)
        # alpha = _extract_into_tensor(self.alpha, t, x.shape)
        # sigma = ddim_eta * self.kappa * th.sqrt(etas_prev / etas) * th.sqrt(alpha)
        # noise = th.randn_like(x)
        
        
        # mean_pred = (
        #     pred_xstart + etas_prev * residual
        #     + th.sqrt(etas_prev*self.kappa**2 - sigma**2) * eps
        # )
        # nonzero_mask = (
        #     (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        # )  # no noise when t == 0
        # sample = mean_pred + nonzero_mask * sigma * noise

        sample = \
            pred_xstart*out["ddim_k"] \
            + out["ddim_m"] * x \
            + out["ddim_j"] * y 
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        y,
        model,
        noise=None,
        first_stage_model=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ddim_eta=0.0,
        zT=None,
        apply_decoder=True,
        one_step=False
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            y=y,
            model=model,
            noise=noise,
            first_stage_model=first_stage_model,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            ddim_eta=ddim_eta,
            zT=zT,
            one_step=one_step
        ):
            final = sample
        if apply_decoder:
            return self.decode_first_stage(final["sample"], first_stage_model)
        return final

    def ddim_sample_loop_progressive(
        self,
        y,
        model,
        noise=None,
        first_stage_model=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ddim_eta=0.0,
        zT=None,
        one_step=False
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        
        if device is None:
            device = next(model.parameters()).device
        z_y = self.encode_first_stage(y, first_stage_model, up_sample=True)
    
        if zT is None:
            z_sample = self.prior_sample(z_y, noise)
        else:
            z_sample = zT

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * z_y.shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model=model,
                    x=z_sample,
                    y=z_y,
                    t=t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    ddim_eta=ddim_eta,
                )
                if one_step:
                    out["sample"]=out["pred_xstart"]
                    yield out
                    break
                yield out
                z_sample = out["sample"]


class GaussianDiffusionDDPM:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarTypeDDPM determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        scale_factor=None,
        sf=4,
    ):
        self.model_mean_type = model_mean_type        # EPSILON
        self.model_var_type = model_var_type          # LEARNED_RANGE
        self.scale_factor = scale_factor   # scale factor in latent space default True
        self.sf=sf

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)

        if self.model_var_type in [ModelVarTypeDDPM.LEARNED, ModelVarTypeDDPM.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarTypeDDPM.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarTypeDDPM.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarTypeDDPM.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:       # predict x_{t-1}
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )                                                  #  predict \eps
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        first_stage_model=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
            
        return self.decode_first_stage(final["sample"], first_stage_model)

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        first_stage_model=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return self.decode_first_stage(final["sample"], first_stage_model)

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device).long()
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, first_stage_model=None, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        z_start = self.encode_first_stage(x_start, first_stage_model)
        if noise is None:
            noise = th.randn_like(z_start)
        z_t = self.q_sample(z_start, t, noise=noise)

        terms = {}

        model_output = model(z_t, t, **model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=z_start, x_t=z_t, t=t
            )[0],
            ModelMeanType.START_X: z_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]
        assert model_output.shape == target.shape == z_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["loss"] = terms["mse"]

        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_zstart = model_output.detach()
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_zstart = self._predict_xstart_from_eps(x_t=z_t, t=t, eps=model_output.detach())
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)   # q(x_t|x_0)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)


    def _scale_input(self, inputs, t):
        return inputs

    def decode_first_stage(self, z_sample, first_stage_model=None):
        ori_dtype = z_sample.dtype
        if first_stage_model is None:
            return z_sample
        else:
            with th.no_grad():
                z_sample = 1 / self.scale_factor * z_sample
                z_sample = z_sample.type(next(first_stage_model.parameters()).dtype)
                out = first_stage_model.decode(z_sample)
                return out.type(ori_dtype)

    def encode_first_stage(self, y, first_stage_model, up_sample=False):
        ori_dtype = y.dtype
        if up_sample:
            y = F.interpolate(y, scale_factor=self.sf, mode='bicubic')
        if first_stage_model is None:
            return y
        else:
            with th.no_grad():
                y = y.type(dtype=next(first_stage_model.parameters()).dtype)
                z_y = first_stage_model.encode(y)
                out = z_y * self.scale_factor
                return out.type(ori_dtype)

