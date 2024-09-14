import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import get_dataloader, RaysData, BaseDataloader
from ray_sampling import RaySampler
from renderer import Renderer
from model import Nerf
import utils


def train_nerf(model: dict, dataloader: BaseDataloader, optimizer, scheduler, criterion, conf):
    resume_epoch, psnr_scores = utils.load_checkpoint(model, optimizer, conf["ckpt_path"])
    renderer = Renderer(conf)
    train_rays = RaysData(*dataloader.get_data(stype="train"))
    val_rays = RaysData(*dataloader.get_data(stype="val"))
    ray_sampler = RaySampler(conf)

    model['nerf'].train()
    pbar = tqdm(total=conf["epochs"])
    pbar.update(resume_epoch)
    for epoch in range(resume_epoch, conf["epochs"]):
        rays_o, rays_d, rays_rgb = train_rays.cast_rays(conf["rays_per_batch"])         # (N_rays, 3), (N_rays, 3), (N_rays, 3)
        points, z_vals = ray_sampler.sample_along_rays(rays_o, rays_d)                  # (N_rays, N_samples, 3), (N_rays, N_samples)
        rays_dn = rays_d.unsqueeze(-2).repeat(1, ray_sampler.N_samples, 1)              # (N_rays, N_samples, 3)

        optimizer.zero_grad()
        rgb, sigmas = model['nerf'](points, rays_dn)                                    # (N_rays, N_samples, 3), (N_rays, N_samples, 1)
        comp_rgb = renderer.volume_render(rgb, sigmas)                                  # (N_rays, 3)
        loss = criterion(comp_rgb, rays_rgb)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: { loss.item() }")
        pbar.update(1)
        if (epoch + 1) % conf["val_interval"] == 0 or epoch == conf["epochs"] - 1:
            model['nerf'].eval()
            with torch.no_grad():
                rays_o, rays_d, rays_rgb = val_rays.cast_image_rays()
                
                # perform batched inference
                comp_rgbs = []
                for (b_rays_o, b_rays_d) in utils.split_batch((rays_o, rays_d), conf["rays_per_batch"]):
                    points, z_vals = ray_sampler.sample_along_rays(b_rays_o, b_rays_d)
                    rays_dn = b_rays_d.unsqueeze(-2).repeat(1, ray_sampler.N_samples, 1)
                
                    rgb, sigmas = model['nerf'](points, rays_dn)
                    comp_rgbs.append(renderer.volume_render(rgb, sigmas))
                comp_rgb = torch.cat(comp_rgbs, dim=0)
                
                curr_psnr = utils.psnr(comp_rgb, rays_rgb)
                print(f"Val psnr: { curr_psnr } dB")
                psnr_scores.append([epoch, curr_psnr])
                # save image
                image = comp_rgb.reshape(val_rays.H, val_rays.W, 3).cpu().numpy()
                # save loss plot
                plt.imsave(f"val_output/{epoch+1}.png", image)
            model['nerf'].train()

        if (epoch + 1) % conf['save_interval'] == 0  or epoch == conf["epochs"] - 1:
            utils.save_checkpoint(epoch + 1, psnr_scores, model, optimizer, conf["ckpt_path"])
            utils.save_psnr_plot(psnr_scores)


def main():
    utils.set_seed(60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = utils.load_yaml("conf.yaml")
    dataloader = get_dataloader(conf["dataset_type"], conf["dataset_path"])
    
    model = {'nerf':Nerf().to(device)}
    optimizer = torch.optim.Adam(model["nerf"].parameters(), lr=conf['lr'])

    final_lr = conf['lr'] / 10
    lr_decay = (final_lr / conf['lr']) ** (conf["lr_decay_interval"]/conf['epochs'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf["lr_decay_interval"], gamma=lr_decay)
    criterion = torch.nn.MSELoss()
    train_nerf(model, dataloader, optimizer, scheduler, criterion, conf)

if __name__ == "__main__":
    main()