import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from dataset.dataset import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import math  # <--- 【新增】引入 math 以計算 Cosine 衰減權重


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=16)

    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='online')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    # @torch.cuda.amp.autocast()
    # def encode(_batch):
    #     return autoencoder.encode(_batch)
    #
    # @torch.cuda.amp.autocast()
    # def decode(_batch):
    #     return autoencoder.decode(_batch)
    @torch.cuda.amp.autocast()
    def encode(_batch):
        # 拆分 RGB (前3通道) 和 Depth (最後1通道)
        rgb = _batch[:, :3, :, :]
        depth = _batch[:, 3:, :, :]

        # 只讓 RGB 通過 autoencoder
        latent_rgb = autoencoder.encode(rgb)

        # 讓 Depth 縮放對齊 latent_rgb 的尺寸
        latent_depth = torch.nn.functional.interpolate(depth, size=latent_rgb.shape[2:], mode='bilinear',
                                                       align_corners=False)

        # 重新黏合回 4 通道
        return torch.cat([latent_rgb, latent_depth], dim=1)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        # 拆分 Latent RGB 和 Latent Depth
        latent_rgb = _batch[:, :-1, :, :]
        latent_depth = _batch[:, -1:, :, :]

        # 只讓 RGB 通過 autoencoder 解碼
        rgb = autoencoder.decode(latent_rgb)

        # 讓 Depth 放大對齊 rgb 的尺寸
        depth = torch.nn.functional.interpolate(latent_depth, size=rgb.shape[2:], mode='bilinear', align_corners=False)

        # 重新黏合回 4 通道
        return torch.cat([rgb, depth], dim=1)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    # set the score_model to train
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())

    def train_step(prime_target, prime_anchor_view, prime_targe_pos, encode_anchor, encode_target):
        _metrics = dict()

        with accelerator.accumulate(nnet):
            if config.train.mode == 'uncond':
                _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
                loss = sde.LSimple(score_model, _z, pred=config.pred)
            elif config.train.mode == 'cond':
                _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
                loss = sde.LSimple(score_model, _z, pred=config.pred, conditions=[encode_anchor, prime_targe_pos])

                # =========================================================================
                # === 【重要新增】：特徵蒸餾 (Feature Distillation) 的 50% 課程學習策略 ===
                # =========================================================================
                # if train_state.step <= config.train.n_steps // 2:
                #     # 取得 Batch 中每張圖各自的特徵 Loss (形狀為 [B])
                #     l_feat_batch = accelerator.unwrap_model(nnet).forward_feature_loss(encode_anchor, prime_targe_pos,
                #                                                                        encode_target)
                #
                #     # 【關鍵防禦】：偵測哪些圖是「非空白」的正常圖片 (排除 CFG dropout 的干擾)
                #     # 判斷方式：如果一張圖的絕對值總和極小 (< 1e-4)，代表它是被捨棄的空白圖
                #     is_active = (prime_anchor_view.abs().view(prime_anchor_view.shape[0], -1).sum(dim=1) > 1e-4)
                #
                #     decay_weight = math.cos(train_state.step / config.train.n_steps * math.pi)
                #
                #     # 只計算那些「正常圖片」的特徵蒸餾 Loss，並把權重降為 0.5 避免喧賓奪主
                #     if is_active.sum() > 0:
                #         valid_l_feat = l_feat_batch[is_active].mean()
                #         loss = loss + 0.5 * valid_l_feat * decay_weight
                #
                #         # 記錄到 wandb
                #         _metrics['l_feat'] = accelerator.gather(valid_l_feat.detach()).mean()
                #     else:
                #         _metrics['l_feat'] = torch.tensor(0.0, device=device)
                # =========================================================================

            else:
                raise NotImplementedError(config.train.mode)

            _metrics['loss'] = accelerator.gather(loss.detach()).mean()
            accelerator.backward(loss.mean())

            optimizer.step()

            if accelerator.sync_gradients:
                lr_scheduler.step()
                train_state.ema_update(config.get('ema_rate', 0.9999))
                train_state.step += 1
                optimizer.zero_grad()

        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        batch = [batch[i].float() for i in range(len(batch))]
        prime_target, prime_anchor_view, prime_targe_pos = batch
        encode_anchor, encode_target = encode(prime_anchor_view), encode(prime_target)
        metrics = train_step(prime_target, prime_anchor_view, prime_targe_pos, encode_anchor, encode_target)

        nnet.eval()

        last_log_step = getattr(train_state, 'last_log_step', -1)
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0 and train_state.step != last_log_step:
            train_state.last_log_step = train_state.step
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        last_grid_step = getattr(train_state, 'last_grid_step', -1)
        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 1 and train_state.step != last_grid_step:
            train_state.last_grid_step = train_state.step
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            z_init = torch.randn(encode_target.size(), device=device)
            if config.train.mode == 'uncond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50)
            elif config.train.mode == 'cond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50,
                                       conditions=[encode_anchor, prime_targe_pos])
            else:
                raise NotImplementedError

            # pred_target = decode(z)
            # pred_target = make_grid(dataset.unpreprocess(pred_target), 10)
            #
            # decode_target = decode(encode_target)
            # decode_target = make_grid(dataset.unpreprocess(decode_target), 10)
            #
            # prime_target = make_grid(dataset.unpreprocess(prime_target), 10)
            # 【關鍵修改】：所有的圖都只取前 3 個通道 (RGB) 去做視覺化儲存
            pred_target = decode(z)[:, :3, :, :]
            pred_target = make_grid(dataset.unpreprocess(pred_target), 10)

            decode_target = decode(encode_target)[:, :3, :, :]
            decode_target = make_grid(dataset.unpreprocess(decode_target), 10)

            prime_target = prime_target[:, :3, :, :]
            prime_target = make_grid(dataset.unpreprocess(prime_target), 10)

            decode_anchor = decode(encode_anchor)[:, :3, :, :]
            decode_anchor = make_grid(dataset.unpreprocess(decode_anchor), 10)

            prime_anchor_view = prime_anchor_view[:, :3, :, :]
            prime_anchor_view = make_grid(dataset.unpreprocess(prime_anchor_view), 10)


            save_image(pred_target, os.path.join(config.sample_dir, f'predict_target-{train_state.step}.png'))
            save_image(decode_target, os.path.join(config.sample_dir, f'decode_target-{train_state.step}.png'))
            save_image(prime_target, os.path.join(config.sample_dir, f'prime_target-{train_state.step}.png'))
            save_image(decode_anchor, os.path.join(config.sample_dir, f'decode_anchor-{train_state.step}.png'))
            save_image(prime_anchor_view, os.path.join(config.sample_dir, f'prime_anchor-{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(pred_target)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        last_eval = getattr(train_state, 'last_eval_step', -1)
        if ((
                    train_state.step % config.train.save_interval == 0 and train_state.step > 0) or train_state.step == config.train.n_steps) and train_state.step != last_eval:
            train_state.last_eval_step = train_state.step
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

            if accelerator.is_main_process:
                import subprocess
                import re
                logging.info(f"開始自動評估第 {train_state.step} 步的模型碼...")

                eval_dir = f"./eval_dir/scenery/1x_step{train_state.step}/"

                eval_cmd = f"torchrun --nproc_per_node=1 evaluate2.py --target_expansion 0.25 0.25 0.25 0.25 --eval_dir {eval_dir} --size 128 --config flickr192_large"
                print(f"正在產圖: {eval_cmd}")
                subprocess.run(eval_cmd, shell=True)

                fid_cmd = f"python -m pytorch_fid {eval_dir}ori/ {eval_dir}gen/"
                print("正在計算 FID...")
                result = subprocess.run(fid_cmd, shell=True, capture_output=True, text=True)

                print(result.stdout)
                if result.stderr:
                    print(result.stderr)

                match = re.search(r"FID:\s+([0-9.]+)", result.stdout)
                if match:
                    fid_score = float(match.group(1))
                    print(f"成功擷取 FID 分數: {fid_score}，準備上傳 wandb!")
                    wandb.log({"eval/FID_1x": fid_score}, step=train_state.step)
                else:
                    print("警告: 無法從輸出中找到 FID 分數。")

    logging.info(f'Finish fitting, step={train_state.step}')
    accelerator.wait_for_everyone()


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'x0pred'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)