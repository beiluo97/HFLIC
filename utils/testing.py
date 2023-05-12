import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.func import image2patch
from utils.metrics import compute_metrics
from utils.utils import *
from utils.func import image2patch, patch2image
from loss.rd_loss import GANLoss
from loss import perceptual_loss as ps


def test_one_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            rec = torch2img(out_net['x_hat'])
            img = torch2img(d)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)

    return loss.avg


def compress_one_image(model, x, stream_path, H, W, img_name):
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    output = os.path.join(stream_path, img_name+"_%04f"%(bpp))
    print(output)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])
    cost_time = out["cost_time"]
    entropy_time = out["entropy_time"]
    encoder_time = out["encoder_time"]
    return bpp, cost_time, entropy_time, encoder_time


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape)

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
    entropy_time = out["entropy_time"]
    decoder_time = out["decoder_time"]
    return x_hat, cost_time, entropy_time, decoder_time


def test_model(test_dataloader, net, logger_test, save_dir, epoch, gpu_id):
    net.eval()
    device = next(net.parameters()).device
    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_bpp = AverageMeter()

    avg_enc_dec_time = AverageMeter()
    avg_enc_time = AverageMeter()
    avg_dec_time = AverageMeter()
    avg_enc_entropy_time = AverageMeter()
    avg_dec_entropy_time = AverageMeter()
    avg_encoder_time = AverageMeter()
    avg_decoder_time = AverageMeter()
    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            B, C, H, W = img.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W

            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            bpp, enc_time, enc_entropy_time, encoder_time = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            x_hat, dec_time, dec_entropy_time, decoder_time = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i))
            rec = torch2img(x_hat)
            img = torch2img(img)
            #img.save(os.path.join(save_dir, '%03d_gt.png' % i))
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            p, m = compute_metrics(rec, img)
            avg_psnr.update(p)
            avg_ms_ssim.update(m)
            avg_bpp.update(bpp)
            avg_enc_time.update(enc_time)
            avg_dec_time.update(dec_time)
            avg_enc_dec_time.update(enc_time + dec_time)
            avg_enc_entropy_time.update(enc_entropy_time)
            avg_dec_entropy_time.update(dec_entropy_time)
            avg_encoder_time.update(encoder_time)
            avg_decoder_time.update(decoder_time)
            logger_test.info(
                f"Image[{i}] | "
                f"Bpp loss: {bpp:.4f} | "
                f"PSNR: {p:.4f} | "
                f"MS-SSIM: {m:.4f} "
                f"Time: {enc_time+dec_time:.4f} | "
                f"Enc Time: {enc_time:.4f} | "
                f"Entropy Enc Time: {enc_entropy_time:.4f} | "
                f"Dec Time: {dec_time:.4f} | "
                f"Entropy dec Time: {dec_entropy_time:.4f} | "
            )
    logger_test.info(
        f"Epoch:[{epoch}] | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr.avg:.4f} | "
        f"Avg MS-SSIM: {avg_ms_ssim.avg:.4f} "

        f"Avg Time: {avg_enc_dec_time.avg:.4f} | "
        f"Avg Enc Time: {avg_enc_time.avg:.4f} | "
        f"Avg Dec Time: {avg_dec_time.avg:.4f} | "
        f"Avg Enc Entropy Time: {avg_enc_entropy_time.avg:.4f} | "
        f"Avg Dec Entropy Time: {avg_dec_entropy_time.avg:.4f} | "
        f"Avg Encoder Time: {avg_encoder_time.avg:.4f} | "
        f"Avg Decoder Time: {avg_decoder_time.avg:.4f} | "            
   )


def test_one_epoch_gan(epoch, test_dataloader, model, model_disc,criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    charbonnier = AverageMeter()
    lpips = AverageMeter()
    style_loss = AverageMeter() 
    adv_loss = AverageMeter() 
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            pred_fake = model_disc(out_net["x_hat"])
            loss_G_fake = gan_loss(pred_fake, False, is_disc=False)
            loss_G_total = (3e-4 * out_criterion["charbonnier"] + 2 * out_criterion["lpips"] + out_criterion["style_loss"] + loss_G_fake +  out_criterion["bpp_loss"])

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            loss.update(loss_G_total.item())
            lpips.update(out_criterion["lpips"].item())
            style_loss.update(out_criterion["style_loss"].item())
            adv_loss.update(loss_G_fake.item())
            if out_criterion["charbonnier"] is not None:
                charbonnier.update(out_criterion["charbonnier"].item())

            rec = torch2img(out_net['x_hat'])
            img = torch2img(d)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: lpips'), lpips.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: style loss'), style_loss.avg, epoch + 1)
    logger_val.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"Charbonnier loss: {charbonnier.avg:.4f} | "
        f"Style loss: {style_loss.avg:.4f} | "
        f"Adv loss: {adv_loss.avg:.4f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f}"
        f"LPIPS: {lpips.avg:.6f}"

    )
    tb_logger.add_scalar('{}'.format('[val]: charbonnier loss'), charbonnier.avg, epoch + 1)
    
    return loss.avg


def test_one_epoch_gan_face(epoch, test_dataloader, model, model_disc,criterion, save_dir, logger_val, tb_logger, config):
    model.eval()
    device = next(model.parameters()).device
    gan_loss = GANLoss('hinge', loss_weight=2.0, real_label_val=1.0, fake_label_val=0.0)

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    charbonnier = AverageMeter()
    lpips = AverageMeter()
    style_loss = AverageMeter() 
    adv_loss = AverageMeter() 
    aux_loss = AverageMeter()
    face_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            mask = d[:, 3:, :, :]   # /255.0     mask_roi
            img = d[:, :3, :, :]    # /255.0
            out_net = model(img)
            out_criterion = criterion(out_net, img, mask)

            pred_fake = model_disc(out_criterion["x_tidle"])
            loss_G_fake = gan_loss(pred_fake, False, is_disc=False)
            loss_G_total = (config["lambda_char"]* out_criterion["charbonnier"] + config["lambda_lpips"] * out_criterion["lpips"] + config["lambda_style"] * out_criterion["style_loss"] + config["lambda_gan"] * loss_G_fake +  out_criterion["bpp_loss"] + config["lambda_face"] * out_criterion["face_loss"])
            
            out_criterion["loss"] =  torch.mean(loss_G_total)
            out_criterion["lpips"] = torch.mean(out_criterion["lpips"])
            out_criterion["face_loss"] = torch.mean(out_criterion["face_loss"])

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            loss.update(loss_G_total.item())
            lpips.update(out_criterion["lpips"].item())
            style_loss.update(out_criterion["style_loss"].item())
            adv_loss.update(loss_G_fake.item())
            face_loss.update(out_criterion["face_loss"].item())
            if out_criterion["charbonnier"] is not None:
                charbonnier.update(out_criterion["charbonnier"].item())

            rec = torch2img(out_net['x_hat'])
            img = torch2img(img)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: lpips'), lpips.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: style loss'), style_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: face loss'), face_loss.avg, epoch + 1)

    logger_val.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"Charbonnier loss: {charbonnier.avg:.4f} | "
        f"Style loss: {style_loss.avg:.4f} | "
        f"Adv loss: {adv_loss.avg:.4f} | "
        f"Face loss: {face_loss.avg:.6f} | "
        f"Bpp loss: {bpp_loss.avg:.4f} | "
        f"Aux loss: {aux_loss.avg:.2f} | "
        f"PSNR: {psnr.avg:.4f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f} | "
        f"LPIPS: {lpips.avg:.6f}"

    )
    tb_logger.add_scalar('{}'.format('[val]: charbonnier loss'), charbonnier.avg, epoch + 1)
    
    return loss.avg
