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


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # mp.set_start_method('spawn')
    # accelerator = accelerate.Accelerator()
    mp.set_start_method('spawn')
    # 假設你的顯卡只能塞 16 的 batch size，你想達到作者的 256，就累積 16 次 (16x16=256)
    # 你可以把 16 換成你需要累積的數字
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
    # assert os.path.exists(dataset.fid_stat)
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

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

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

        # 加上這行 with，讓 accelerate 自動幫我們處理梯度累積
        with accelerator.accumulate(nnet):
            # optimizer.zero_grad()

            # 計算 Loss (跟原本一樣)
            if config.train.mode == 'uncond':
                _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
                loss = sde.LSimple(score_model, _z, pred=config.pred)
            elif config.train.mode == 'cond':
                _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
                loss = sde.LSimple(score_model, _z, pred=config.pred, conditions=[encode_anchor, prime_targe_pos])
            else:
                raise NotImplementedError(config.train.mode)

            _metrics['loss'] = accelerator.gather(loss.detach()).mean()
            accelerator.backward(loss.mean())

            # 這裡 accelerate 會自動判斷：如果累積次數還沒到，就不會執行 step()
            optimizer.step()
            # lr_scheduler.step()

            # 【重要修改】只有在累積滿了、真正執行參數更新的那一步，才更新步數和 EMA
            if accelerator.sync_gradients:
                lr_scheduler.step()
                train_state.ema_update(config.get('ema_rate', 0.9999))
                train_state.step += 1

                optimizer.zero_grad()

        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)
    # def train_step(prime_target, prime_anchor_view, prime_targe_pos, encode_anchor, encode_target):
    #
    #     _metrics = dict()
    #     optimizer.zero_grad()
    #     if config.train.mode == 'uncond':
    #         _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
    #         loss = sde.LSimple(score_model, _z, pred=config.pred)
    #     elif config.train.mode == 'cond':
    #         _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
    #         loss = sde.LSimple(score_model, _z, pred=config.pred, conditions=[encode_anchor, prime_targe_pos])
    #     else:
    #         raise NotImplementedError(config.train.mode)
    #     _metrics['loss'] = accelerator.gather(loss.detach()).mean()
    #     accelerator.backward(loss.mean())
    #     optimizer.step()
    #     lr_scheduler.step()
    #     train_state.ema_update(config.get('ema_rate', 0.9999))
    #     train_state.step += 1
    #     return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)
    #
    # logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        batch = [batch[i].float() for i in range(len(batch))]
        prime_target, prime_anchor_view, prime_targe_pos = batch
        encode_anchor, encode_target = encode(prime_anchor_view), encode(prime_target)
        metrics = train_step(prime_target, prime_anchor_view, prime_targe_pos, encode_anchor, encode_target)

        nnet.eval()
        # if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
        #     logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
        #     logging.info(config.workdir)
        #     wandb.log(metrics, step=train_state.step)

        last_log_step = getattr(train_state, 'last_log_step', -1)

        # 2. 加上防連點條件：這步還沒印過 log 才准進去
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0 and train_state.step != last_log_step:
            # 3. 馬上上鎖！
            train_state.last_log_step = train_state.step

            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        # if accelerator.is_main_process and train_state.step % config.train.eval_interval == 1:
        last_grid_step = getattr(train_state, 'last_grid_step', -1)

        # 2. 加上防連點條件：確保這步還沒畫過圖
        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 1 and train_state.step != last_grid_step:
            # 3. 馬上把現在這步記錄下來上鎖！
            train_state.last_grid_step = train_state.step
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            z_init = torch.randn(encode_target.size(), device=device)
            if config.train.mode == 'uncond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50)
            elif config.train.mode == 'cond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50, conditions=[encode_anchor, prime_targe_pos])
            else:
                raise NotImplementedError

            # through diffusion
            pred_target = decode(z)
            pred_target = make_grid(dataset.unpreprocess(pred_target), 10)

            # through autoencoder
            decode_target = decode(encode_target)
            decode_target = make_grid(dataset.unpreprocess(decode_target), 10)

            # ground truth
            prime_target = make_grid(dataset.unpreprocess(prime_target), 10)

            # through autoencoder
            decode_anchor = decode(encode_anchor)
            decode_anchor = make_grid(dataset.unpreprocess(decode_anchor), 10)

            # prime condition
            prime_anchor_view = make_grid(dataset.unpreprocess(prime_anchor_view), 10)

            save_image(pred_target, os.path.join(config.sample_dir, f'predict_target-{train_state.step}.png'))
            save_image(decode_target, os.path.join(config.sample_dir, f'decode_target-{train_state.step}.png'))
            save_image(prime_target, os.path.join(config.sample_dir, f'prime_target-{train_state.step}.png'))
            save_image(decode_anchor, os.path.join(config.sample_dir, f'decode_anchor-{train_state.step}.png'))
            save_image(prime_anchor_view, os.path.join(config.sample_dir, f'prime_anchor-{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(pred_target)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        # if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
        #     torch.cuda.empty_cache()
        #     logging.info(f'Save and eval checkpoint {train_state.step}...')
        #     if accelerator.local_process_index == 0:
        #         train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
        #     accelerator.wait_for_everyone()
        #     torch.cuda.empty_cache()
        # accelerator.wait_for_everyone()
        last_eval = getattr(train_state, 'last_eval_step', -1)

        # 2. 加入判斷：必須符合倍數，而且「這個 step 還沒被評估過」
        if ((train_state.step % config.train.save_interval == 0 and train_state.step > 0) or train_state.step == config.train.n_steps) and train_state.step != last_eval:
            train_state.last_eval_step = train_state.step
        # if (train_state.step % config.train.save_interval == 0 and train_state.step > 0) or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

            # ========== 加入以下自動評估與上傳 W&B 程式碼 ==========
            if accelerator.is_main_process:
                import subprocess
                import re  # 用來擷取字串中的分數
                logging.info(f"開始自動評估第 {train_state.step} 步的模型碼...")

                # 建立一個專屬這次 step 的資料夾
                eval_dir = f"./eval_dir/scenery/1x_step{train_state.step}/"

                # 1. 執行 Evaluate 產圖
                eval_cmd = f"torchrun --nproc_per_node=1 evaluate2.py --target_expansion 0.25 0.25 0.25 0.25 --eval_dir {eval_dir} --size 128 --config flickr192_large"
                print(f"正在產圖: {eval_cmd}")
                subprocess.run(eval_cmd, shell=True)

                # 2. 自動算 FID，並加入 capture_output=True 來攔截終端機輸出
                fid_cmd = f"python -m pytorch_fid {eval_dir}ori/ {eval_dir}gen/"
                print("正在計算 FID...")
                result = subprocess.run(fid_cmd, shell=True, capture_output=True, text=True)

                # 把原本應該印在終端機的字手動印出來，方便你檢查
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)

                # 3. 從 stdout 中擷取 FID 分數並上傳 wandb
                # 通常 pytorch_fid 最後一行會印出 "FID:  39.89..."
                match = re.search(r"FID:\s+([0-9.]+)", result.stdout)
                if match:
                    fid_score = float(match.group(1))
                    print(f"成功擷取 FID 分數: {fid_score}，準備上傳 wandb!")

                    # 記錄到 wandb，你可以自己取喜歡的圖表名稱，例如 "eval/FID_1x"
                    wandb.log({"eval/FID_1x": fid_score}, step=train_state.step)
                else:
                    print("警告: 無法從輸出中找到 FID 分數。")
            # ======================================================

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
