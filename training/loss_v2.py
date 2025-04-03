# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, split_dims, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.split_dims = split_dims

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.

                # print('gen_img shape:', gen_img.shape)

                img, mask = gen_img.split(self.split_dims, dim=1)

                # print('img shape:', img.shape)
                # print('mask shape:', mask.shape)

                img_fg = img * mask
                img_bg = img * (1.0 - mask)

                gen_logits_fg = self.run_D(img_fg, gen_c, sync=False)
                gen_logits_bg = self.run_D(img_bg, gen_c, sync=False)

                # logits are the output of the discriminator, which is a measure of how real or fake the image is
                # the discriminator is trained to output a high value for real images and a low value for fake images
                training_stats.report('Loss/scores/fake_fg', gen_logits_fg)
                training_stats.report('Loss/scores/fake_bg', gen_logits_bg)
                training_stats.report('Loss/signs/fake', gen_logits_fg.sign())
                training_stats.report('Loss/signs/fake', gen_logits_bg.sign())

                loss_Gmain = 0.5*torch.nn.functional.softplus(-gen_logits_fg) + 0.5*torch.nn.functional.softplus(-gen_logits_bg) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)

                img, mask = gen_img.split(self.split_dims, dim=1)

                img_fg = img * mask
                img_bg = img * (1.0 - mask)

                gen_logits_fg = self.run_D(img_fg, gen_c, sync=False)
                gen_logits_bg = self.run_D(img_bg, gen_c, sync=False)

                training_stats.report('Loss/scores/fake_fg', gen_logits_fg)
                training_stats.report('Loss/scores/fake_bg', gen_logits_bg)
                training_stats.report('Loss/signs/fake', gen_logits_fg.sign())
                training_stats.report('Loss/signs/fake', gen_logits_bg.sign())

                loss_Dgen = 0.5*torch.nn.functional.softplus(gen_logits_fg) + 0.5*torch.nn.functional.softplus(gen_logits_bg)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)

                # print('real_img_tmp shape:', real_img_tmp.shape)

                img, mask = real_img_tmp.split(self.split_dims, dim=1)

                img_fg = img * mask
                img_bg = img * (1.0 - mask)

                real_logits_fg = self.run_D(img_fg, real_c, sync=sync)
                real_logits_bg = self.run_D(img_bg, real_c, sync=sync)

                training_stats.report('Loss/scores/real_fg', real_logits_fg)
                training_stats.report('Loss/scores/real_bg', real_logits_bg)
                training_stats.report('Loss/signs/real_fg', real_logits_fg.sign())
                training_stats.report('Loss/signs/real_bg', real_logits_bg.sign())
                
                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = 0.5*torch.nn.functional.softplus(-real_logits_fg) + 0.5*torch.nn.functional.softplus(-real_logits_bg) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0

                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        img, mask = real_img_tmp.split(self.split_dims, dim=1)

                        img_fg = img * mask
                        img_bg = img * (1.0 - mask)

                        real_logits_fg = self.run_D(img_fg, real_c, sync=sync)
                        real_logits_bg = self.run_D(img_bg, real_c, sync=sync)

                        r1_grads = \
                            torch.autograd.grad(outputs=[real_logits_fg.sum()], inputs=[img_fg],
                                                create_graph=True, only_inputs=True)[0]
                        r1_grads2 = \
                            torch.autograd.grad(outputs=[real_logits_bg.sum()], inputs=[img_bg],
                                                create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    r1_penalty2 = r1_grads2.square().sum([1, 2, 3])
                    loss_Dr1 = 0.5* r1_penalty * (self.r1_gamma / 2) + 0.5* r1_penalty2 * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/r1_penalty2', r1_penalty2)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------